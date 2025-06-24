library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)
library(readr)
library(data.table)

df <- read_csv("weights_all.csv")

df <- df %>%
  rename(recap_date = `Date at recapture`)

df <- df %>%
  rename(recap_mass = `Mass at Recapture`)

df <- df %>%
  rename(trip_dur = `Avg. Trip Duration`)

df <- df %>%
  rename(Maternal_Habitat = `Maternal Foraging Habitat`)

# Ensure your date columns are in Date or POSIXct format
df$Date <- as.Date(df$Date, format = "%m/%d/%Y")
df$recap_date <- as.Date(df$recap_date, format = "%m/%d/%Y")


# Calculate growth rate (e.g., in kg per day)
df$Growth_Rate <- (df$recap_mass - df$ActualMass) / as.numeric(df$recap_date - df$Date)


iso <- fread("ISOTOPES-ALL.csv", encoding = "UTF-8")


df_joined <- df %>%
  left_join(iso, by = "RightTagNumber")

#write.csv(df_joined, "weightiso_data.csv", row.names = FALSE)
#########################################################################################
library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)
library(readr)
library(data.table)
df_joined <- fread("weightiso_data.csv")

# Filter for Age A and complete cases
df_adults <- df_joined %>%
  filter(Age == "A")

df_p <- df_joined %>%
  filter(Age == "P")


ggplot(df_p, aes(x = LE_carbon, y = Nitrogen, color = Rookery, fill = Rookery)) +
  geom_point(size = 3, shape = 21, alpha = 0.8) +
  stat_ellipse(geom = "polygon", alpha = 0.2, linetype = "dashed") +
  labs(
    title = "δ15N vs. δ13C (LE) for Pup NFS",
    x = expression("Serum"~ delta^13*C~"(LE)"),
    y = expression("Serum"~ delta^15*N),
    color = "Rookery",
    fill = "Rookery"
  ) +
  # Manual color and fill scales - customize these colors as needed
  scale_color_manual(values = c("Vostochni" = "#E31A1C", "Reef" = "#1F78B4")) +
  scale_fill_manual(values = c("Vostochni" = "#E31A1C", "Reef" = "#1F78B4")) +
  theme_minimal(base_size = 14)
  theme_minimal(base_size = 14)
  
  

  # Ensure known/unknown are separated
  df_known <- df_p[!is.na(df_p$Maternal_Habitat2), ]
  df_all <- df_p  # includes all pups
  
ggplot() +
    # All pups: background layer (no shape)
  geom_point(data = df_all, aes(x = LE_carbon, y = Nitrogen, 
                                color = Rookery, fill = Rookery),
              size = 2.5, shape = 21, alpha = 0.4) +
    
    # Only pups with known maternal habitat: shape overlay
  geom_point(data = df_known, aes(x = LE_carbon, y = Nitrogen, 
                                  color = Rookery, fill = Rookery, 
                                  shape = Maternal_Habitat2),
              size = 3.2, alpha = 0.9) +
    
    # Ellipses for Rookery groups
  stat_ellipse(data = df_all, aes(x = LE_carbon, y = Nitrogen, 
                                  group = Rookery, fill = Rookery), 
                geom = "polygon", alpha = 0.2, linetype = "dashed") +
    
  labs(
      title = "δ15N vs. δ13C (LE) for Pup NFS",
      x = expression("Serum"~ delta^13*C~"(LE)"),
      y = expression("Serum"~ delta^15*N),
      color = "Rookery",
      fill = "Rookery",
      shape = "Maternal Habitat"
    ) +
  scale_color_manual(values = c("Vostochni" = "#E31A1C", "Reef" = "#1F78B4")) +
  scale_fill_manual(values = c("Vostochni" = "#E31A1C", "Reef" = "#1F78B4")) +
  scale_shape_manual(values = c("On-shelf dominant" = 22, "Off-shelf dominant" = 24)) +

  theme_minimal(base_size = 14)
  
  

################

df_p_ <- df_p %>%
    filter(!is.na(Rookery), !is.na(LE_carbon), !is.na(Nitrogen))
summary(lm(LE_carbon ~ Rookery, data = df_p_))   # For δ13C
summary(lm(Nitrogen ~ Rookery, data = df_p_))    # For δ15N
# Make sure both variables are numeric and Rookery is a factor
df_p$Rookery <- as.factor(df_p$Rookery)

# MANOVA model
manova_model <- manova(cbind(LE_carbon, Nitrogen) ~ Rookery, data = df_p_)
summary(manova_model, test = "Pillai")

  


ggplot(df_p, aes(x = LE_carbon, y = Nitrogen)) +
  # Pup points colored by rookery
  geom_point(aes(fill = Rookery), color = "black", shape = 21, size = 3, alpha = 0.8) +
  
  # Rookery ellipses (shaded)
  stat_ellipse(aes(color = Rookery, fill = Rookery), geom = "polygon", alpha = 0.2, linetype = "dashed") +
  
  # Maternal habitat ellipses (outline only)
  stat_ellipse(
    data = df_p,
    aes(color = Maternal_Habitat),
    linetype = "dotdash",
    fill = NA,
    size = 1
  ) +
  
  labs(
    title = "δ15N vs. δ13C (LE) for Pup NFS\nwith Maternal Habitat Overlay",
    x = expression(delta^13*C~"(LE)"),
    y = expression(delta^15*N),
    fill = "Rookery",
    color = "Rookery / Maternal Habitat"
  ) +
  theme_minimal(base_size = 14)

# Plot of all adult females by foraging habitat

######## adults rookery #############
ggplot(df_adults, aes(x = LE_carbon, y = Nitrogen)) +
  # Adult points colored by location (habitat) with different shapes
  # Purple outline for VOSTOCHNI rookery adults
  # Create combined habitat categories
  geom_point(aes(fill = ifelse(location %in% c("Off", "Mixed"), "Off-shelf dominant", "On-shelf dominant"), 
                 shape = ifelse(location %in% c("Off", "Mixed"), "Off-shelf dominant", "On-shelf dominant"),
                 color = ifelse(Rookery == "Vostochni", "Vostochni", "Reef")), 
             size = 3, alpha = 0.8, stroke = 1.0) +
  
  # Rookery ellipses (shaded) - you can uncomment if needed
  #stat_ellipse(aes(color = Rookery, fill = Rookery), geom = "polygon", alpha = 0.2, linetype = "dashed") +
  
  # Define custom shapes for combined habitat categories
  scale_shape_manual(values = c("Off-shelf dominant" = 22,        # square (fillable)
                                "On-shelf dominant" = 21)) + # circle (fillable)
  
  # Define custom colors for combined habitat categories
  scale_fill_manual(values = c("Off-shelf dominant" = "coral",
                               "On-shelf dominant" = "lightblue")) +
  
  # Remove the color scale for ellipse since we're using direct color
  # scale_color_manual(values = c("VOSTOCHNI" = "purple")) +
  
  # Custom color scale for point outlines
  scale_color_manual(values = c("Vostochni" = "purple", "Reef" = "black"),
                     name = "Rookery") +
  
  labs(
    title = "δ15N vs. δ13C (LE) for Adult NFS\nwith Foraging Habitat Overlay",
    x = expression("Plasma"~ delta^13*C~"(LE)"),
    y = expression("Plasma"~ delta^15*N),
    fill = "Foraging Habitat",
    shape = "Foraging Habitat",
    color = "Rookery"
  ) +
  theme_minimal(base_size = 14)
##################################

# Only for ellipse calculation — drop rows with NA in ellipse vars
df_p_ellipse <- df_p %>%
  filter(!is.na(Maternal_Habitat), !is.na(LE_carbon), !is.na(Nitrogen))

ggplot(df_p, aes(x = LE_carbon, y = Nitrogen)) +
  # Show all pup points (even ones not used in ellipses)
  geom_point(aes(fill = Maternal_Habitat), color = "black", shape = 21, size = 3, alpha = 0.8) +
  
  # Only use complete data for ellipses
  stat_ellipse(
    data = df_p_ellipse,
    aes(color = Maternal_Habitat, fill = Maternal_Habitat),
    geom = "polygon",
    alpha = 0.2,
    linetype = "dashed"
  ) +
  
  labs(
    title = "δ15N vs. δ13C (LE) in Pup Blood\nColored by Maternal Foraging Habitat",
    x = expression(delta^13*C~"(LE)"),
    y = expression(delta^15*N),
    fill = "Maternal Habitat",
    color = "Maternal Habitat"
  ) +
  theme_minimal(base_size = 14)

################################################################


# Check how many adults you have in each category
cat("Summary of adult females by foraging habitat:\n")
print(table(df_adults$Maternal_Habitat, useNA = "ifany"))

cat("\nTotal adult females:", nrow(df_adults), "\n")
cat("Adult females with complete isotope + habitat data:", nrow(df_adults_ellipse), "\n")


###############################################################
# CARBON #

# Calculate R² value
lm_model <- lm(ppLE_carbon ~ LE_carbon + pp_Sex, data = df_adults)
r_squared <- summary(lm_model)$r.squared
summary(lm_model)

p1 <- ggplot(df_adults, aes(x = LE_carbon, y = ppLE_carbon)) +
  # Points colored by maternal foraging habitat with different shapes
  # Purple outline for VOSTOCHNI rookery adults
  geom_point(aes(fill = ifelse(location %in% c("Off", "Mixed"), "Off-shelf dominant", "On-shelf dominant"), 
                 shape = ifelse(location %in% c("Off", "Mixed"), "Off-shelf dominant", "On-shelf dominant"),
                 color = ifelse(Rookery == "Vostochni", "Vostochni", "Reef")), 
             size = 3, alpha = 0.8, stroke = 1.5) +
  
  # Add 1:1 reference line that extends through the data range
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50", size = 0.8) +
  
  # Alternative: add a smooth trend line to see the actual relationship
  geom_smooth(method = "lm", se = FALSE, color = "gray50", linetype = "solid") +
  
  
  
  # Define custom shapes for combined habitat categories
  scale_shape_manual(values = c("Off-shelf dominant" = 22,        # square (fillable)
                                "On-shelf dominant" = 21)) + # circle (fillable)
  
  # Define custom colors for combined habitat categories
  scale_fill_manual(values = c("Off-shelf dominant" = "black",
                               "On-shelf dominant" = "black")) +
  
  # Custom color scale for point outlines
  scale_color_manual(values = c("Vostochni" = "coral", "Reef" = "black"),
                     name = "Rookery") +
  
  # Fix the legend appearance
  guides(
    fill = guide_legend(title = "Maternal Foraging Habitat"),
    shape = guide_legend(title = "Maternal Foraging Habitat"),
    color = guide_legend(title = "Rookery", 
                         override.aes = list(shape = 21, 
                                             fill = "white",
                                             size = 3))
  ) +
  
  labs(
    #title = "Maternal vs. Pup δ13C (LE) Comparison",
    x = expression("Maternal"~delta^13*C~"(LE)"),
    y = expression("Pup"~delta^13*C~"(LE)"),
    fill = "Maternal Foraging Habitat",
    shape = "Maternal Foraging Habitat",
    color = "Rookery"
  ) +
  theme_minimal(base_size = 14) +
  # Make axes equal for better comparison
  coord_equal()

# Print additional model statistics
cat("Linear Model Results:\n")
cat(sprintf("R² = %.3f\n", r_squared))
cat(sprintf("Adjusted R² = %.3f\n", summary(lm_model)$adj.r.squared))
cat(sprintf("p-value = %.4f\n", summary(lm_model)$coefficients[2,4]))
cat(sprintf("Slope = %.3f\n", coef(lm_model)[2]))
cat(sprintf("Intercept = %.3f\n", coef(lm_model)[1]))

## NITROGEN ##

lm_model <- lm(ppNitrogen ~ Nitrogen + pp_Sex, data = df_adults)
r_squared <- summary(lm_model)$r.squared
summary(lm_model)

p2 <- ggplot(df_adults, aes(x = Nitrogen, y = ppNitrogen)) +
  # Points colored by maternal foraging habitat with different shapes
  # Purple outline for VOSTOCHNI rookery adults
  geom_point(aes(fill = ifelse(location %in% c("Off", "Mixed"), "Off-shelf dominant", "On-shelf dominant"), 
                 shape = ifelse(location %in% c("Off", "Mixed"), "Off-shelf dominant", "On-shelf dominant"),
                 color = ifelse(Rookery == "Vostochni", "Vostochni", "Reef")), 
             size = 3, alpha = 0.8, stroke = 1.5) +
  
  # Add 1:1 reference line that extends through the data range
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50", size = 0.8) +
  
  # Alternative: add a smooth trend line to see the actual relationship
  geom_smooth(method = "lm", se = FALSE, color = "gray50", linetype = "solid") +
  
  # Define custom shapes for combined habitat categories
  scale_shape_manual(values = c("Off-shelf dominant" = 22,        # square (fillable)
                                "On-shelf dominant" = 21)) + # circle (fillable)
  
  # Define custom colors for combined habitat categories
  scale_fill_manual(values = c("Off-shelf dominant" = "black",
                               "On-shelf dominant" = "black")) +
  
  # Custom color scale for point outlines
  scale_color_manual(values = c("Vostochni" = "coral", "Reef" = "black"),
                     name = "Rookery") +
  
  # Fix the legend appearance
  guides(
    fill = guide_legend(title = "Maternal Foraging Habitat"),
    shape = guide_legend(title = "Maternal Foraging Habitat"),
    color = guide_legend(title = "Rookery", 
                         override.aes = list(shape = 21, 
                                             fill = "white",
                                             size = 3))
  ) +
  
  labs(
    #title = "Maternal vs. Pup δ15N Comparison",
    x = expression("Maternal"~delta^15*N),
    y = expression("Pup"~delta^15*N),
    fill = "Maternal Foraging Habitat",
    shape = "Maternal Foraging Habitat",
    color = "Rookery"
  ) +
  theme_minimal(base_size = 14) +
  # Make axes equal for better comparison
  coord_equal()

# Print additional model statistics
cat("Linear Model Results:\n")
cat(sprintf("R² = %.3f\n", r_squared))
cat(sprintf("Adjusted R² = %.3f\n", summary(lm_model)$adj.r.squared))
cat(sprintf("p-value = %.4f\n", summary(lm_model)$coefficients[2,4]))
cat(sprintf("Slope = %.3f\n", coef(lm_model)[2]))
cat(sprintf("Intercept = %.3f\n", coef(lm_model)[1]))

# Combine plots using patchwork
library(patchwork)
combined_plot <- p1 + p2 + plot_layout(guides = "collect")

# Display the combined plot
dev.new(width = 24, height = 12)
print(combined_plot)

ggsave("combined_isotope_plots.png", combined_plot, 
       width = 14, height =6, dpi = 300, units = "in")

#############################################
# GROWTH RATES#
lm_model <- lm(ppGrowth_Rate ~ trip_dur, data = df_adults)
r_squared <- summary(lm_model)$r.squared

ggplot(df_adults, aes(x = trip_dur, y = ppGrowth_Rate)) +
  # Points colored by maternal foraging habitat with different shapes
  # Purple outline for VOSTOCHNI rookery adults
  geom_point(aes(fill = ifelse(location %in% c("Off", "Mixed"), "Off-shelf dominant", "On-shelf dominant"), 
                 shape = ifelse(location %in% c("Off", "Mixed"), "Off-shelf dominant", "On-shelf dominant"),
                 color = ifelse(Rookery == "Vostochni", "Vostochni", "Reef")), 
             size = 3, alpha = 0.8, stroke = 1.5) +
  
  # Add a smooth trend line to see the relationship
  geom_smooth(method = "lm", se = FALSE, color = "gray50", linetype="dashed", alpha = 0.3) +
  
  # Define custom shapes for combined habitat categories
  scale_shape_manual(values = c("Off-shelf dominant" = 22,        # square (fillable)
                                "On-shelf dominant" = 21)) + # circle (fillable)
  
  # Define custom colors for combined habitat categories
  scale_fill_manual(values = c("Off-shelf dominant" = "black",
                               "On-shelf dominant" = "blue")) +
  
  # Custom color scale for point outlines
  scale_color_manual(values = c("Vostochni" = "blue", "Reef" = "black"),
   name = "Rookery") +
                     
  # Fix the legend appearance
  guides(
    fill = guide_legend(title = "Maternal Foraging Habitat"),
    shape = guide_legend(title = "Maternal Foraging Habitat"),
    color = guide_legend(title = "Rookery", 
                         override.aes = list(shape = 21, 
                                             fill = "white",
                                             size = 3))
  ) +
                    
  
  labs(
    title = "Maternal Trip Duration vs. Paired Pup Growth Rate",
    x = "Trip Duration (days)",
    y = "Pup Growth Rate (g/day)",
    fill = "Maternal Foraging Habitat",
    shape = "Maternal Foraging Habitat",
    color = "Rookery"
  ) +
  theme_minimal(base_size = 14)+xlim(6, 10)

cat("Linear Model Results:\n")
cat(sprintf("R² = %.3f\n", r_squared))
cat(sprintf("Adjusted R² = %.3f\n", summary(lm_model)$adj.r.squared))
cat(sprintf("p-value = %.4f\n", summary(lm_model)$coefficients[2,4]))
cat(sprintf("Slope = %.3f\n", coef(lm_model)[2]))
cat(sprintf("Intercept = %.3f\n", coef(lm_model)[1]))

########################################################################
# GR model #

library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)
library(readr)
library(data.table)

df <- read.csv("weightiso_data.csv")

df2 <- read.csv("pup_behavior_summary.csv")
df2$AccelId <- sub("_.*", "", df2$Pup_ID)


# Filter for Age A and complete cases
df_adults <- df %>%
  filter(Age == "A")

df_p <- df %>%
  filter(Age == "P")

pups <- df_p %>%
  left_join(df2, by = "AccelId")
pups <- pups[pups$Pup_ID != "UU_0000_S2", ]
pups <- pups[pups$Pup_ID != "W_0000_S2", ] # 0 growth rate

# Keep only rows with complete data for the model
df_p_clean <- pups %>%
  filter(!is.na(Growth_Rate) & !is.na(LE_carbon) & !is.na(Nitrogen) & !is.na(Rookery) & !is.na(Sex)
         & !is.na(Assumed_MH) & !is.na(Nursing_Flag_Avg_Daily_Frequency_Percent) & !is.na(Nursing_Flag_Total_Bouts)
         & !is.na(Nursing_Flag_Avg_Daily_Duration_Minutes) & !is.na(Nursing_Flag_Avg_Bout_Duration_Minutes) 
         & !is.na(Nursing_Flag_Max_Bout_Duration_Minutes) & !is.na(Nursing_Flag_Avg_Daily_Bout_Count))

# Then re-run the model selection on clean data
library(MuMIn)
options(na.action = "na.fail")

global_model <- lm(Growth_Rate ~ LE_carbon + Nitrogen + Rookery + Sex + Assumed_MH  +
                     Nursing_Flag_Avg_Daily_Bout_Count + Nursing_Flag_Avg_Daily_Frequency_Percent, data = df_p_clean)

global_model2 <- lm(Growth_Rate ~ Rookery + Sex + Assumed_MH + 
                      Nursing_Flag_Avg_Daily_Frequency_Percent, data = df_p_clean)

global_model3 <- lm(Growth_Rate ~ Rookery + Sex + Assumed_MH + 
                      Nursing_Flag_Avg_Daily_Frequency_Percent + Nursing_Flag_Total_Bouts +
                    Nursing_Flag_Avg_Daily_Duration_Minutes + Nursing_Flag_Avg_Bout_Duration_Minutes +
                      Nursing_Flag_Max_Bout_Duration_Minutes +
                      Nursing_Flag_Avg_Daily_Bout_Count, data = df_p_clean)

global_model4 <- lm(Growth_Rate ~  Nitrogen + LE_Carbon, data = df_p_clean)

global_model5 <- global_model <- lm(trip_dur ~  location, data = df_adults)

summary(global_model)
summary(global_model2)
summary(global_model3)
summary(global_model4)
summary(global_model5)
#dredge
model_set <- dredge(global_model)

# View model selection table
head(model_set)

# Best model (lowest AICc)
best_model <- get.models(model_set, 1)[[1]]
summary(best_model)

# Model averaging for models with delta AICc < 2
avg_model <- model.avg(model_set, subset = delta < 2)
summary(avg_model)

#xxx and GR
ggplot(df_p_clean, aes(x = Nursing_Flag_Avg_Daily_Bout_Count, y = Growth_Rate, color = Rookery)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "gray50", fill = "lightgray", linewidth = 1.0) +
  scale_color_manual(values = c("black", "red")) +  # Assuming first level is not Vostochni
  labs(
    title = "Effect of Nursing bouts on Pup Growth Rate",
    x = "Average Daily Nursing Bouts (Count)",
    y = "Growth Rate (g/day)"
  ) +
  theme_minimal(base_size = 14)

# Plot

df_p_clean3 <- pups %>%
  filter(!is.na(Growth_Rate) & 
          !is.na(Assumed_MH))

ggplot(df_p_clean3, aes(x = Assumed_MH, y = Growth_Rate, fill = Assumed_MH)) +
  geom_boxplot(alpha = 0.6) +
  geom_jitter(width = 0.1, size = 2) +
  labs(title = "Pup Growth Rate by Maternal Foraging Habitat",
       y = "Growth Rate (g/day)",
       x = "Maternal Habitat") +
  theme_minimal()

model <- lm(Growth_Rate ~ Nursing_Flag_Avg_Daily_Bout_Count, data = df_p_clean)
summary(model)


############################################################
# IGNORE _ Code not used

df_p_clean2 <- df_p %>%
  filter(!is.na(Growth_Rate) & !is.na(LE_carbon) & !is.na(Nitrogen) & !is.na(Maternal_Habitat2) & !is.na(Sex))
growth_model_habitat <- lm(Growth_Rate ~ Nitrogen + LE_carbon + Maternal_Habitat2 +Sex, data = df_p_clean2)
summary(growth_model_habitat)
growth_model_habitat_int <- lm(Growth_Rate ~ Nitrogen * Maternal_Habitat2 + LE_carbon * Maternal_Habitat2, data = df_p_clean2)
summary(growth_model_habitat_int)

##dredge - model select #
options(na.action = "na.fail")

global_model <- lm(Growth_Rate ~ LE_carbon * Maternal_Habitat2 + Nitrogen * Maternal_Habitat2, data = df_p_clean2)
model_set <- dredge(global_model)

# View model selection table
head(model_set)

# Best model (lowest AICc)
best_model <- get.models(model_set, 1)[[1]]
summary(best_model)

# Model averaging for models with delta AICc < 2
avg_model <- model.avg(model_set, subset = delta < 2)
summary(avg_model)

habitat_model <- lm(Growth_Rate ~ Maternal_Habitat2, data = df_p_clean2)
summary(habitat_model)

# Plot
ggplot(df_p_clean2, aes(x = Maternal_Habitat2, y = Growth_Rate, fill = Maternal_Habitat2)) +
  geom_boxplot(alpha = 0.6) +
  geom_jitter(width = 0.1, size = 2) +
  labs(title = "Pup Growth Rate by Maternal Foraging Habitat",
       y = "Growth Rate (g/day)",
       x = "Maternal Habitat") +
  theme_minimal()


###########################################################
# 
# # Load required library
# library(broom)
# 
# # Build the growth rate model
# # Model 1: Full model with interactions
# growth_model_full <- lm(Growth_Rate ~ LE_carbon * Rookery * Sex + Nitrogen * Rookery * Sex, 
#                         data = df_p_clean)
# 
# # Model 2: Additive model (no interactions)
# growth_model_additive <- lm(Growth_Rate ~ LE_carbon + Nitrogen + Rookery + Sex, 
#                             data = df_p_clean)
# 
# # Model 3: Simple model with just rookery
# growth_model_rookery <- lm(Growth_Rate ~ Rookery, 
#                            data = df_p_clean)
# 
# # Compare models using AIC
# model_comparison <- data.frame(
#   Model = c("Full (with interactions)", "Additive", "Rookery only"),
#   AIC = c(AIC(growth_model_full), AIC(growth_model_additive), AIC(growth_model_rookery)),
#   R_squared = c(summary(growth_model_full)$r.squared, 
#                 summary(growth_model_additive)$r.squared,
#                 summary(growth_model_rookery)$r.squared)
# )
# 
# # Display model comparison
# print("Model Comparison:")
# print(model_comparison)
# 
# # Summary of best model (lowest AIC)
# best_model <- growth_model_additive  # You can change this based on AIC results
# print("\nBest Model Summary:")
# summary(best_model)
# 
# # ANOVA to test significance of rookery effect
# print("\nANOVA - Testing Rookery Effect:")
# anova(growth_model_additive)
# 
# # Post-hoc pairwise comparisons between rookeries (if significant)
# if(length(unique(df_p$Rookery)) > 1) {
#   #install.packages("emmeans")
#   library(emmeans)
#   rookery_means <- emmeans(best_model, ~ Rookery)
#   print("\nRookery Means:")
#   print(rookery_means)
#   
#   print("\nPairwise Comparisons:")
#   print(pairs(rookery_means))
# }
# 
# # Create prediction plot
# 
# # Predict values for plotting
# # Only add predictions and residuals for rows with complete data
# complete_cases <- complete.cases(df_p[, c("Growth_Rate", "LE_carbon", "Nitrogen", "Rookery")])
# df_p$predicted <- NA
# df_p$residuals <- NA
# df_p$predicted[complete_cases] <- predict(best_model)
# df_p$residuals[complete_cases] <- residuals(best_model)
# 
# # Plot observed vs predicted
# ggplot(df_p, aes(x = predicted, y = Growth_Rate)) +
#   geom_point(aes(color = Rookery, shape = Rookery), size = 3, alpha = 0.7) +
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
#   labs(title = "Observed vs Predicted Pup Growth Rate",
#        x = "Predicted Growth Rate (g/day)",
#        y = "Observed Growth Rate (g/day)",
#        color = "Rookery",
#        shape = "Rookery") +
#   theme_minimal() +
#   coord_equal()
# 
# # Residual plot
# ggplot(df_p, aes(x = predicted, y = residuals)) +
#   geom_point(aes(color = Rookery), alpha = 0.7) +
#   geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
#   labs(title = "Residuals vs Fitted Values",
#        x = "Fitted Values",
#        y = "Residuals") +
#   theme_minimal()
# 
# 
# ####################################################################
# ## Including nursing duration now ##
# ####################################################################
# 
# library(ggplot2)
# library(dplyr)
# library(tidyr)
# library(lubridate)
# library(readr)
# library(data.table)
# library(ggplot2)
# 
# df <- read.csv("weightiso_data.csv")
# 
# df2 <- read.csv("pup_nursing_summary.csv")
# df2$AccelId <- sub("_.*", "", df2$Pup_ID)
# 
# 
# # Filter for Age A and complete cases
# df_adults <- df %>%
#   filter(Age == "A")
# 
# df_p <- df %>%
#   filter(Age == "P")
# 
# pups <- df_p %>%
#   left_join(df2, by = "AccelId")
# pups <- pups[pups$Pup_ID != "UU_0000_S2", ]
# 
# # visualize
# 
# ggplot(pups, aes(x = Avg_Daily_Nursing_Frequency_Percent, y = Growth_Rate)) +
#   geom_point() +
#   geom_smooth(method = "lm") +
#   labs(title = "Growth Rate vs Nursing Frequency")
# 
# ggplot(pups, aes(x = Avg_Daily_Nursing_Duration_Minutes, y = Growth_Rate)) +
#   geom_point() +
#   geom_smooth(method = "lm") +
#   labs(title = "Growth Rate vs Nursing Duration")
# 
# ggplot(pups, aes(x = Total_Nursing_Bouts, y = Growth_Rate)) +
#   geom_point() +
#   geom_smooth(method = "lm") +
#   labs(title = "Growth Rate vs Total Nursing Bouts")
# 
# ggplot(pups, aes(x = Avg_Bout_Duration_Minutes, y = Growth_Rate)) +
#   geom_point() +
#   geom_smooth(method = "lm") +
#   labs(title = "Growth Rate vs Nursing Duration")
# 
# pups <- pups %>% 
#   dplyr::select(Growth_Rate, Avg_Daily_Nursing_Frequency_Percent, 
#                 Avg_Daily_Nursing_Duration_Minutes, Total_Nursing_Bouts, 
#                 Avg_Bout_Duration_Minutes) %>%
#   na.omit()
# 
# # now stats!
# model_freq <- lm(Growth_Rate ~ Avg_Daily_Nursing_Frequency_Percent, data = pups)
# model_dur <- lm(Growth_Rate ~ Total_Nursing_Bouts, data = pups)
# model_both <- lm(Growth_Rate ~ Avg_Daily_Nursing_Frequency_Percent + Total_Nursing_Bouts, data = pups)
# model_int <- lm(Growth_Rate ~ Avg_Daily_Nursing_Frequency_Percent * Total_Nursing_Bouts, data = pups)
# summary(model_freq)
# summary(model_dur)
# summary(model_both)
# summary(model_int)
# 
# # Dredge
# library(MuMIn)
# global_model <- lm(Growth_Rate ~ Avg_Daily_Nursing_Frequency_Percent + 
#                      Avg_Daily_Nursing_Duration_Minutes + Total_Nursing_Bouts + 
#                      Avg_Bout_Duration_Minutes, data = pups, na.action = na.fail)
# model_set <- dredge(global_model)
# head(model_set)  # Shows top-ranked models by AICc
# best_model <- get.models(model_set, 1)[[1]]
# summary(best_model)
# #optional if multiple good AICc models
# avg_model <- model.avg(model_set, subset = delta < 2)
# summary(avg_model)
# 
# ## including more than just nursing
# 
# pups <- df_p %>%
#   left_join(df2, by = "AccelId")
# pups <- pups[pups$Pup_ID != "UU_0000_S2", ]
# 
# lm_model <- lm(Growth_Rate ~ Rookery + Maternal_Habitat2 + Avg_Daily_Nursing_Frequency_Percent, 
#                data = pups)
# summary(lm_model)
# 
# #dredge
# pups <- pups %>% 
#   dplyr::select(Growth_Rate, Avg_Daily_Nursing_Frequency_Percent, 
#                 Avg_Daily_Nursing_Duration_Minutes, Total_Nursing_Bouts, 
#                 Avg_Bout_Duration_Minutes, ActualMass, Maternal_Habitat2, Rookery) %>%
#   na.omit()
# global_model <- lm(Growth_Rate ~ Rookery + Maternal_Habitat2 + 
#                      Avg_Daily_Nursing_Frequency_Percent +
#                      Avg_Daily_Nursing_Duration_Minutes + 
#                      ActualMass,
#                    data = pups, na.action = na.fail)
# model_set <- dredge(global_model)
# head(model_set)  # Shows top-ranked models by AICc
# best_model <- get.models(model_set, 1)[[1]]
# summary(best_model)
# #optional if multiple good AICc models
# avg_model <- model.avg(model_set, subset = delta < 2)
# summary(avg_model)
# 
# ### no nursing is significant with growth rate!

### all behaviors ###

library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)
library(readr)
library(data.table)

df <- read.csv("weightiso_data.csv")

df2 <- read.csv("pup_behavior_summary.csv")
df2$AccelId <- sub("_.*", "", df2$Pup_ID)


# Filter for Age A and complete cases
df_adults <- df %>%
  filter(Age == "A")

df_p <- df %>%
  filter(Age == "P")

pups <- df_p %>%
  left_join(df2, by = "AccelId")
pups <- pups[pups$Pup_ID != "UU_0000_S2", ]


df_p_clean2 <- pups %>%
  filter(!is.na(Growth_Rate) & !is.na(LE_carbon) & !is.na(Nitrogen) & !is.na(Assumed_MH) & !is.na(Sex) & !is.na(Rookery)
         & !is.na(ActualMass) & !is.na(Swimming_Flag_Avg_Daily_Frequency_Percent) & !is.na(Resting_Flag_Avg_Daily_Frequency_Percent)
         & !is.na(High_Activity_Flag_Avg_Daily_Frequency_Percent) & !is.na(Nursing_Flag_Avg_Daily_Frequency_Percent))

# Fit multivariate linear model
fit <- lm(High_Activity_Flag_Avg_Daily_Frequency_Percent ~ Rookery + Assumed_MH + ActualMass + Growth_Rate + Sex, data = df_p_clean2)
fit2 <- lm(Swimming_Flag_Avg_Daily_Frequency_Percent ~ Growth_Rate + Sex + Assumed_MH, data = df_p_clean2)
fit3 <- lm(recap_mass ~ High_Activity_Flag_Avg_Daily_Frequency_Percent, data = df_p_clean2)

model_set <- dredge(fit3)
head(model_set)  # Shows top-ranked models by AICc
best_model <- get.models(model_set, 1)[[1]]
summary(best_model)
#optional if multiple good AICc models
avg_model <- model.avg(model_set, subset = delta < 2)
summary(avg_model)

# Summary of individual regressions
summary(fit2)


# # Select behavior frequency outcomes
# Y <- pups[, c("Swimming_Flag_Avg_Daily_Frequency_Percent",
#               "Nursing_Flag_Avg_Daily_Frequency_Percent",
#               "Resting_Flag_Avg_Daily_Frequency_Percent",
#               "High_Activity_Flag_Avg_Daily_Frequency_Percent")]
# 
# # MANOVA-style test of overall effects
# manova_fit <- manova(as.matrix(Y) ~ Rookery + Assumed_MH + ActualMass + Growth_Rate, data = pups)
# summary(manova_fit)
# summary(manova_fit, test = "Pillai")  # robust to non-normality
# 
# lm(High_Activity_Flag_Avg_Daily_Frequency_Percent ~ Rookery * Maternal_Habitat2 + ActualMass + Growth_Rate, data=pups)
# car::leveneTest(High_Activity_Flag_Avg_Daily_Frequency_Percent ~ Rookery, data = pups)
# car::leveneTest(High_Activity_Flag_Avg_Daily_Frequency_Percent ~ Maternal_Habitat2, data = pups)
# manova_fit <- manova(as.matrix(Y) ~ Rookery * Maternal_Habitat2 + ActualMass + Growth_Rate, data = pups)
# summary(manova_fit, test = "Pillai")

library(ggplot2)

ggplot(df_p_clean2, aes(x = Assumed_MH, y = Swimming_Flag_Avg_Daily_Frequency_Percent, fill = Assumed_MH)) +
  geom_boxplot(alpha = 0.7) +
  labs(y = "Avg Daily Nursing Frequency (%)", x = "Maternal Habitat Type") +
  theme_minimal()

ggplot(df_p_clean2, aes(x = Rookery, y = High_Activity_Flag_Avg_Daily_Frequency_Percent, fill = Rookery)) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  labs(y = "High_activity (%)", x = "Rookery") +
  theme_classic()

ggplot(aes(x = recap_mass, y = Nursing_Flag_Avg_Daily_Duration_Minutes), data = df_p_clean2) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(y = "Avg Daily Nursing (Minutes)", x = "Final Mass at Recapture (kg)") +
  theme_classic()

ggplot(df_p_clean2, aes(x = Sex, y = recap_mass, fill = Sex)) +
  geom_boxplot(position = position_dodge(width = 0.75)) +
  labs(y = "Mass", x = "Sex") +
  theme_classic()



library(tidyr)
p_long <- df_p_clean2 %>%
  pivot_longer(cols = ends_with("Frequency_Percent"),
               names_to = "Behavior", values_to = "Percent")

ggplot(p_long, aes(x = Assumed_MH, y = Percent, fill = Behavior)) +
  geom_boxplot(alpha = 0.8) +
  facet_wrap(~ Behavior, scales = "free_y") +
  theme_bw() +
  labs(x = "Maternal Habitat", y = "Avg Daily Frequency (%)")

# Example: boxplot of daily swimming frequency by rookery
df_p_clean2 %>%
  filter(!is.na(Assumed_MH), !is.na(Nursing_Flag_Avg_Daily_Frequency_Percent)) %>%
  ggplot(aes(x = Assumed_MH, y = Nursing_Flag_Avg_Daily_Frequency_Percent)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Swimming Frequency by Maternal Habitat",
       x = "Maternal Habitat",
       y = "Avg Daily Nursing Frequency (%)")

# Example: nursing duration by maternal habitat
df_p_clean2 %>%
  filter(!is.na(Rookery), !is.na(Nursing_Flag_Avg_Daily_Frequency_Percent)) %>%
ggplot(aes(x = Rookery, y = Nursing_Flag_Avg_Daily_Frequency_Percent)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Nursing Duration by Rookery",
       y = "Avg Daily Nursing Duration (min)")

library(tidyr)
library(dplyr)


# Create linear models for each behavior
swimming_model <- lm(Swimming_Flag_Avg_Daily_Frequency_Percent ~ 
                       Rookery + Assumed_MH + Growth_Rate + recap_mass, 
                     data = df_p_clean2)

nursing_model <- lm(Nursing_Flag_Avg_Daily_Frequency_Percent ~ 
                      Rookery + Assumed_MH + Growth_Rate + recap_mass, 
                    data = df_p_clean2)

resting_model <- lm(Resting_Flag_Avg_Daily_Frequency_Percent ~ 
                      Rookery + Assumed_MH + Growth_Rate + recap_mass, 
                    data = df_p_clean2)

high_activity_model <- lm(High_Activity_Flag_Avg_Daily_Frequency_Percent ~ 
                            Rookery + Assumed_MH + Growth_Rate + recap_mass, 
                          data = df_p_clean2)

# Check each model
summary(swimming_model)
summary(nursing_model)
summary(resting_model)
summary(high_activity_model)

###################
# MANOVA #

library(car)

# Create response matrix
behavior_matrix <- cbind(df_p_clean2$Swimming_Flag_Avg_Daily_Frequency_Percent,
                         df_p_clean2$Nursing_Flag_Avg_Daily_Frequency_Percent,
                         df_p_clean2$Resting_Flag_Avg_Daily_Frequency_Percent,
                         df_p_clean2$High_Activity_Flag_Avg_Daily_Frequency_Percent)

# MANOVA model
manova_model <- manova(behavior_matrix ~ Rookery + Assumed_MH + Growth_Rate + recap_mass, 
                       data = df_p_clean2)

summary(manova_model, test = "Wilks")
summary.aov(manova_model)  # Univariate tests

####################################
behavior_df <- df_p_clean2 %>%
  select(Pup_ID, Assumed_MH,
         Swimming = Swimming_Flag_Avg_Daily_Frequency_Percent,
         Nursing = Nursing_Flag_Avg_Daily_Duration_Minutes,
         Resting = Resting_Flag_Avg_Daily_Frequency_Percent,
         High_Activity = High_Activity_Flag_Avg_Daily_Frequency_Percent) %>%
  pivot_longer(cols = c(Swimming, Nursing, Resting, High_Activity),
               names_to = "Behavior", values_to = "Value")


library(ggpubr)
behavior_df %>%
  filter(!is.na(Assumed_MH)) %>%
  ggplot(aes(x = Assumed_MH, y = Value, fill = Assumed_MH)) +
  geom_boxplot() +
  stat_compare_means(method = "t.test", 
                     label = "p.format", 
                     label.y.npc = 0.9) +  # Position of p-value
  scale_fill_brewer(palette = "Set2") +
  facet_wrap(~ Behavior, scales = "free_y") +
  labs(title = "Pup Behaviors by Maternal Foraging Habitat", y = "Value") +
  theme_classic()
###########################################
behavior_df <- df_p_clean2 %>%
  select(Pup_ID, Rookery,
         Swimming = Swimming_Flag_Avg_Daily_Frequency_Percent,
         Nursing = Nursing_Flag_Avg_Daily_Frequency_Percent,
         Resting = Resting_Flag_Avg_Daily_Frequency_Percent,
         High_Activity = High_Activity_Flag_Avg_Daily_Frequency_Percent) %>%
  pivot_longer(cols = c(Swimming, Nursing, Resting, High_Activity),
               names_to = "Behavior", values_to = "Value")

library(ggpubr)
behavior_df %>%
  filter(!is.na(Rookery)) %>%
  ggplot(aes(x = Rookery, y = Value, fill = Rookery)) +
  geom_boxplot() +
  stat_compare_means(method = "t.test", 
                     label = "p.format", 
                     label.y.npc = 0.9) +
  scale_fill_brewer(palette = "Set2") +
  facet_wrap(~ Behavior, scales = "free_y") +
  labs(title = "Pup Behaviors by Rookery", y = "Average Daily Time (%)") +
  theme_classic() +
  theme(
    axis.text.x = element_text(size = 14),        # X-axis tick labels
    axis.text.y = element_text(size = 14),        # Y-axis tick labels
    axis.title.x = element_text(size = 16),       # X-axis title
    axis.title.y = element_text(size = 16),       # Y-axis title
    plot.title = element_text(size = 18),         # Main title
    strip.text = element_text(size = 14),         # Facet panel labels
    legend.text = element_text(size = 12),        # Legend text
    legend.title = element_text(size = 14)        # Legend title
  )
###########################################
behavior_df <- df_p_clean2 %>%
  select(Pup_ID, Sex,
         Swimming = Swimming_Flag_Avg_Daily_Frequency_Percent,
         Nursing = Nursing_Flag_Avg_Daily_Frequency_Percent,
         Resting = Resting_Flag_Avg_Daily_Frequency_Percent,
         High_Activity = High_Activity_Flag_Avg_Daily_Frequency_Percent) %>%
  pivot_longer(cols = c(Swimming, Nursing, Resting, High_Activity),
               names_to = "Behavior", values_to = "Value")

library(ggpubr)
behavior_df %>%
  filter(!is.na(Sex)) %>%
  ggplot(aes(x = Sex, y = Value, fill = Sex)) +
  geom_boxplot() +
  stat_compare_means(method = "t.test", 
                     label = "p.format",  
                     label.y.npc = 0.9) +  # Position of p-value
  scale_fill_brewer(palette = "Set2") +
  facet_wrap(~ Behavior, scales = "free_y") +
  labs(title = "Pup Behaviors by Sex", y = "Value") +
  theme_classic()
#########################################################

# Basic linear model
nursing_mass_model <- lm(Nursing_Flag_Avg_Daily_Frequency_Percent ~ recap_mass + Rookery, 
                         data = df_p_clean2)
nursing_mass_model <- lm(Growth_Rate ~ Sex, 
                         data = df_p_clean2)
summary(nursing_mass_model)

# Visualize the relationship
library(ggplot2)
df_p_clean2 %>%
  ggplot(aes(x = recap_mass, y = Nursing_Flag_Avg_Daily_Frequency_Percent)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +  # Add regression line with confidence interval
  labs(x = "Pup mass at recapture (kg)", 
       y = "Average Daily Nursing (%)",
       title = "Relationship Between Pup Mass and Nursing Frequency") +
  theme_classic()

# Check model assumptions
par(mfrow = c(2,2))
plot(nursing_mass_model)
par(mfrow = c(1,1))




