# ## SPLINES - GAMMs ##


# # Penalized Spline GAM, Cyclic Cubic Spline and Thin Plate Cyclic Spline GAMs for Time Series Analysis in R
# ################### and lots of comparisons for which is best ################

# ###################################################
# #### not averaging - using all data in 30 minute bins ####
# ##################################################

# # Penalized Spline GAM, Cyclic Cubic Spline and Thin Plate Cyclic Spline GAMs 
# # Using Individual Observations (not averaged proportions)

# ################################################################
# # ADDING P SPLINES AND CONFIDENCE INTERVALS AND ALL DATA NOT AVERAGING #
# ################################################################
# library(mgcv)
# library(nlme)
# library(ggplot2)
# library(dplyr)
# library(tidyr)
# library(lubridate)
# library(readr)
# library(gridExtra)
# library(grid)
# library(AICcmodavg)
# library(knitr)  # For table output

# # Load and preprocess
# output <- read_csv("RawF_predictions_with_probs.csv") # change accelID
# output$Timestamp <- mdy_hms(output$Timestamp)

# output$hour <- as.numeric(format(output$Timestamp, "%H")) + as.numeric(format(output$Timestamp, "%M")) / 60

# # Map numeric behavior to labels
# behaviors <- c(
#   "-1" = "Uncertain",
#   "0" = "Resting",
#   "1" = "Nursing",
#   "2" = "High_activity",
#   "3" = "Inactive",
#   "4" = "Swimming"
# )
# output$behavior_label <- behaviors[as.character(output$Filtered_Predicted_Behavior)]

# # Create binary indicators for each behavior (using ALL individual data points)
# output_expanded <- output %>%
#   filter(!is.na(behavior_label)) %>%
#   select(hour, behavior_label) %>%
#   mutate(
#     is_resting = as.numeric(behavior_label == "Resting"),
#     is_nursing = as.numeric(behavior_label == "Nursing"), 
#     is_high_activity = as.numeric(behavior_label == "High_activity"),
#     is_swimming = as.numeric(behavior_label == "Swimming"),
#     is_inactive = as.numeric(behavior_label == "Inactive"),
#     is_uncertain = as.numeric(behavior_label == "Uncertain")
#   )

# # Also create binned data for comparison plots
# output$hour_bin <- floor(output$hour * 2) / 2
# prop_data <- output %>%
#   group_by(hour_bin, behavior_label) %>%
#   summarise(count = n(), .groups = "drop") %>%
#   group_by(hour_bin) %>%
#   mutate(total = sum(count), prop = count / total) %>%
#   ungroup()

# # GAM fitting using ALL individual data points with binomial family
# model_list <- list()
# aicc_table <- data.frame()

# selected_behaviors <- c("Resting", "Swimming", "Nursing", "High_activity")
# behavior_columns <- c("is_resting", "is_swimming", "is_nursing", "is_high_activity")
# names(behavior_columns) <- selected_behaviors

# output_expanded$hour_cos <- cos(2 * pi * output_expanded$hour / 24)
# output_expanded$hour_sin <- sin(2 * pi * output_expanded$hour / 24)

# for (i in seq_along(selected_behaviors)) {
#   b <- selected_behaviors[i]
#   col_name <- behavior_columns[i]
  
#   if (sum(output_expanded[[col_name]]) < 50) next  # Need sufficient positive cases
  
#   # Cyclic cubic spline
#   m_cc <- gam(as.formula(paste(col_name, "~ s(hour, bs = 'cc', k = 6)")), 
#               data = output_expanded, family = binomial, method = "REML", select = TRUE,
#               knots = list(hour = c(0, 24)))
  
#   # Penalized regression spline  
#   m_cp <- gam(as.formula(paste(col_name, "~ s(hour, bs = 'cp', k = 6)")), 
#               data = output_expanded, family = binomial, method = "REML", select = TRUE)
  
#   # Circular thin plate spline
#   m_tp <- gam(as.formula(paste(col_name, "~ s(hour_cos, hour_sin, bs = 'tp', k = 6)")),
#               data = output_expanded, family = binomial, method = "REML", select = TRUE)

  
  
#   # Store models
#   model_list[[paste(b, "cc")]] <- m_cc
#   model_list[[paste(b, "cp")]] <- m_cp
#   model_list[[paste(b, "tp")]] <- m_tp
  
#   # Calculate AICc
#   aicc_table <- rbind(aicc_table, data.frame(
#     behavior = b,
#     spline = c("cc", "cp", "tp"),
#     AICc = c(AICc(m_cc), AICc(m_cp), AICc(m_tp))
#   ))
# }

# # Select best models
# best_models <- aicc_table %>%
#   group_by(behavior) %>%
#   slice_min(order_by = AICc, n = 1) %>%
#   ungroup()

# # Predict for best models with confidence intervals
# plot_data <- data.frame()
# new_hours <- data.frame(hour = seq(0, 23.99, length.out = 500))
# new_hours$hour_cos <- cos(2 * pi * new_hours$hour / 24)
# new_hours$hour_sin <- sin(2 * pi * new_hours$hour / 24)


# for (i in 1:nrow(best_models)) {
#   b <- best_models$behavior[i]
#   spline_type <- best_models$spline[i]
#   mod <- model_list[[paste(b, spline_type)]]
  
#   # Get predictions with standard errors (on link scale)
#   preds <- predict(mod, newdata = new_hours, se.fit = TRUE)
  
#   # Convert from logit scale to probability scale
#   predicted_prob <- plogis(preds$fit)
#   lower_prob <- plogis(preds$fit - 1.96 * preds$se.fit)
#   upper_prob <- plogis(preds$fit + 1.96 * preds$se.fit)
  
#   plot_data <- rbind(plot_data, data.frame(
#     hour = new_hours$hour,
#     predicted = predicted_prob,
#     lower = lower_prob,
#     upper = upper_prob,
#     behavior = b,
#     spline = spline_type
#   ))
# }

# # Calculate confidence intervals
# # (Already calculated above in the prediction loop)

# # Prepare all spline plots with confidence intervals
# plot_data_all <- data.frame()

# for (b in selected_behaviors) {
#   col_name <- behavior_columns[b]
#   if (sum(output_expanded[[col_name]]) < 50) next
  
#   for (sp in c("cc", "cp", "tp")) {
#     mod <- model_list[[paste(b, sp)]]
#     if (is.null(mod)) next
    
#     preds <- predict(mod, newdata = new_hours, se.fit = TRUE)
    
#     # Convert from logit scale to probability scale
#     predicted_prob <- plogis(preds$fit)
#     lower_prob <- plogis(preds$fit - 1.96 * preds$se.fit)
#     upper_prob <- plogis(preds$fit + 1.96 * preds$se.fit)
    
#     plot_data_all <- rbind(plot_data_all, data.frame(
#       hour = new_hours$hour,
#       predicted = predicted_prob,
#       lower = lower_prob,
#       upper = upper_prob,
#       behavior = b,
#       spline = sp
#     ))
#   }
# }

# #check EDF
# gam.check(m_cc)
# gam.check(m_cp)
# gam.check(m_tp)

# # Spline comparison plot
# plot1 = ggplot(plot_data_all, aes(x = hour, y = predicted, color = spline)) +
#   geom_line(size = 1) +
#   geom_ribbon(aes(ymin = lower, ymax = upper, fill = spline), alpha = 0.2, color = NA) +
#   facet_wrap(~ behavior, scales = "free_y") +
#   labs(x = "Hour of Day", y = "Proportion of Time", 
#        title = "Spline Types Compared per Behavior (with 95% CI)") +
#   theme_classic() +
#   scale_x_continuous(breaks = seq(0, 24, by = 4)) +
#   scale_color_manual(values = c("cc" = "blue", "cp" = "red", "tp" = "green"),
#                      labels = c("Cyclic Cubic", "Penalized", "Thin Plate")) +
#   scale_fill_manual(values = c("cc" = "blue", "cp" = "red", "tp" = "green"),
#                     labels = c("Cyclic Cubic", "Penalized", "Thin Plate"))

# # Filter and plot best models for selected behaviors with confidence intervals
# plot_data_sel <- plot_data %>% filter(behavior %in% selected_behaviors)

# # Multi-color plot with confidence intervals
# plot2= ggplot(plot_data_sel, aes(x = hour, y = predicted, color = behavior, fill = behavior)) +
#   geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
#   geom_line(size = 1) +
#   labs(x = "Hour of Day", y = "Proportion of Time", 
#        title = "Best GAM Spline per Behavior with 95% Confidence Intervals") +
#   theme_classic() +
#   scale_x_continuous(breaks = seq(0, 24, by = 2))

# # Individual faceted plots with confidence intervals
# plot3 = ggplot(plot_data_sel, aes(x = hour, y = predicted)) +
#   geom_ribbon(aes(ymin = lower, ymax = upper), fill = "steelblue", alpha = 0.3) +
#   geom_line(color = "steelblue", size = 1) +
#   facet_wrap(~ behavior, scales = "free_y") +
#   labs(x = "Hour of Day", y = "Proportion of Time", 
#        title = "Best GAM Spline per Behavior with 95% Confidence Intervals") +
#   theme_classic() +
#   scale_x_continuous(breaks = seq(0, 24, by = 4))

# # Enhanced plot showing observed data points with confidence intervals
# # Use binned data for visualization (to avoid overplotting thousands of points)
# obs_data <- prop_data %>% filter(behavior_label %in% selected_behaviors)

# plot4 = ggplot() +
#   geom_point(data = obs_data, aes(x = hour_bin, y = prop), 
#              alpha = 0.6, size = 1.5, color = "gray50") +
#   geom_ribbon(data = plot_data_sel, aes(x = hour, ymin = lower, ymax = upper), 
#               fill = "steelblue", alpha = 0.3) +
#   geom_line(data = plot_data_sel, aes(x = hour, y = predicted), 
#             color = "steelblue", size = 1.2) +
#   facet_wrap(~ behavior, scales = "free_y") +
#   labs(x = "Hour of Day", y = "Proportion of Time", 
#        title = "GAM Fits using ALL Individual Data Points",
#        subtitle = "Gray points = binned observed data (for visualization), Blue line = GAM fitted to all raw data, Shaded area = 95% CI") +
#   theme_classic() +
#   scale_x_continuous(breaks = seq(0, 24, by = 4))

# # Additional plot showing sample size per hour to demonstrate data density
# sample_size_data <- output_expanded %>%
#   mutate(hour_bin = floor(hour * 2) / 2) %>%
#   group_by(hour_bin) %>%
#   summarise(n_observations = n(), .groups = "drop")

# plot5 = ggplot(sample_size_data, aes(x = hour_bin, y = n_observations)) +
#   geom_col(fill = "steelblue", alpha = 0.7) +
#   labs(x = "Hour of Day", y = "Number of Observations", 
#        title = "Data Density: Number of Individual Observations per 30-minute Bin",
#        subtitle = "Shows the actual amount of raw data used to fit the GAMs") +
#   theme_classic() +
#   scale_x_continuous(breaks = seq(0, 24, by = 2))

# # Show enhanced AICc table
# cat("### AICc Table for GAM Splines per Behavior\n")
# aicc_summary <- aicc_table %>%
#   arrange(behavior, AICc) %>%
#   mutate(spline_label = case_when(
#     spline == "cc" ~ "Cyclic Cubic",
#     spline == "cp" ~ "Penalized",
#     spline == "tp" ~ "Thin Plate"
#   ))

# print(kable(aicc_summary %>% select(behavior, spline_label, AICc), 
#             col.names = c("Behavior", "Spline Type", "AICc"), digits = 2))

# # Show best models summary
# cat("\n### Best Models Selected (Lowest AICc)\n")
# best_summary <- best_models %>%
#   mutate(spline_label = case_when(
#     spline == "cc" ~ "Cyclic Cubic",
#     spline == "cp" ~ "Penalized", 
#     spline == "tp" ~ "Thin Plate"
#   ))

# print(kable(best_summary %>% select(behavior, spline_label, AICc),
#             col.names = c("Behavior", "Best Spline Type", "AICc"), digits = 2))


# ggsave("best_spline_F.png", plot = plot1, width = 10, height = 6, dpi = 300)
# ggsave("best_spline_DS.png", plot = plot1, width = 10, height = 6, dpi = 300)

# ggsave("best_spline_behavior_F.png", plot = plot2, width = 10, height = 6, dpi = 300)
# ggsave("best_spline_behavior_DS.png", plot = plot2, width = 10, height = 6, dpi = 300)

# ggsave("best_gam_behaviors_F.png", plot = plot3, width = 10, height = 6, dpi = 300)
# ggsave("best_gam_behaviors_DS.png", plot = plot3, width = 10, height = 6, dpi = 300)

# ggsave("gam_fits.png", plot = plot4, width = 10, height = 4, dpi = 300)

# ggsave("data_density_plot.png", plot = plot5, width = 10, height = 4, dpi = 300)


### Add random effects for all pups and autoregression ###
## they are all overfit/underfit/but idk how to fix ##


#######################################################################################
# ALL PUPS - random spline! include auoregression too #
#######################################################################################

# Thin Plate Circular GAM Analysis
# Population-level activity budget using thin plate splines on circular coordinates
library(mgcv)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)
library(readr)
library(knitr)
library(gridExtra)

# Load and preprocess
output <- read_csv("allpups_predictions_final.csv") 
output$Timestamp <- mdy_hms(output$Timestamp)
output$hour <- as.numeric(format(output$Timestamp, "%H")) + as.numeric(format(output$Timestamp, "%M")) / 60

# Create time ordering
output <- output %>%
  arrange(Tag.ID, Timestamp) %>%
  group_by(Tag.ID) %>%
  mutate(time_order = row_number()) %>%
  ungroup()

# Map behaviors
behaviors <- c(
  "-1" = "Uncertain",
  "0" = "Resting", 
  "1" = "Nursing",
  "2" = "High_activity",
  "3" = "Inactive",
  "4" = "Swimming"
)
output$behavior_label <- behaviors[as.character(output$Filtered_Predicted_Behavior)]

# Create binary indicators
output_expanded <- output %>%
  filter(!is.na(behavior_label)) %>%
  select(Tag.ID, hour, time_order, behavior_label, Timestamp) %>%
  mutate(
    is_resting = as.numeric(behavior_label == "Resting"),
    is_nursing = as.numeric(behavior_label == "Nursing"), 
    is_high_activity = as.numeric(behavior_label == "High_activity"),
    is_swimming = as.numeric(behavior_label == "Swimming"),
    pup_id = as.factor(Tag.ID),
    hour_cos = cos(2 * pi * hour / 24),
    hour_sin = sin(2 * pi * hour / 24)
  )

# THIN PLATE CIRCULAR GAMM MODELS
model_list <- list()
selected_behaviors <- c("Resting", "Swimming", "Nursing", "High_activity")
behavior_columns <- c("is_resting", "is_swimming", "is_nursing", "is_high_activity")
names(behavior_columns) <- selected_behaviors

cat("=== FITTING THIN PLATE CIRCULAR MODELS ===\n")

for (i in seq_along(selected_behaviors)) {
  b <- selected_behaviors[i]
  col_name <- behavior_columns[i]
  
  behavior_count <- sum(output_expanded[[col_name]], na.rm = TRUE)
  cat(paste("\n--- Processing", b, "behavior ---\n"))
  cat(paste("Total occurrences:", behavior_count, "\n"))
  
  if (behavior_count < 100) {
    cat(paste("Skipping", b, "- insufficient data\n"))
    next
  }
  
  # Fit thin plate circular GAM with random intercepts
  cat("Fitting thin plate circular GAMM...\n")
  
  m_tp <- tryCatch({
    gam(as.formula(paste(col_name, "~ s(hour_cos, hour_sin, bs = 'tp', k = 8) + s(pup_id, bs = 're')")),
        data = output_expanded,
        family = binomial,
        method = "REML")
  }, error = function(e) {
    cat("Model failed:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(m_tp)) {
    model_list[[b]] <- m_tp
    cat("Model fitted successfully\n")
    
    # Quick model summary
    cat("AIC:", AIC(m_tp), "\n")
    cat("Deviance explained:", round(summary(m_tp)$dev.expl * 100, 1), "%\n")
  }
}

# SAVE MODELS AND DATA FOR LATER USE
save(model_list, output_expanded, file = "gam_models_and_data.RData")
cat("\n=== MODELS AND DATA SAVED TO 'gam_models_and_data.RData' ===\n")

#################################################################
# if reloading
# Load required libraries
library(ggplot2)
library(gridExtra)  # This was missing!
library(mgcv)

# Load your saved models and data
load('gam_models_and_data.RData')

# Check what was loaded
cat("Loaded objects:\n")
print(ls())

# Check what pup_ids are available in the original data
cat("Available pup_ids:\n")
if("output_expanded" %in% ls()) {
  available_pups <- unique(output_expanded$pup_id)
  print(head(available_pups, 10))
  # Use the first pup_id as reference (it will be excluded anyway)
  reference_pup <- available_pups[1]
} else {
  # If output_expanded not available, use a generic ID
  reference_pup <- "PUP001"  # Adjust this if needed
}

# Create prediction data (24-hour cycle) with trigonometric transformations
hours_seq <- seq(0, 24, by = 0.1)
new_hours <- data.frame(
  hour = hours_seq,
  hour_cos = cos(2 * pi * hours_seq / 24),
  hour_sin = sin(2 * pi * hours_seq / 24),
  pup_id = reference_pup  # Add pup_id (will be excluded in predictions)
)

# Initialize empty data frame for plot data
plot_data <- data.frame()

# Generate predictions for each behavior
for (b in names(model_list)) {
  mod <- model_list[[b]]
  
  # POPULATION-LEVEL PREDICTION (exclude random effects)
  preds <- predict(mod, newdata = new_hours, se.fit = TRUE, exclude = "s(pup_id)")
  
  # Convert to probability scale
  predicted_prob <- plogis(preds$fit)
  lower_prob <- plogis(preds$fit - 1.96 * preds$se.fit)
  upper_prob <- plogis(preds$fit + 1.96 * preds$se.fit)
  
  plot_data <- rbind(plot_data, data.frame(
    hour = new_hours$hour,
    predicted = predicted_prob,
    lower = lower_prob,
    upper = upper_prob,
    behavior = b
  ))
}

# Define colors with nursing highlighted
behavior_colors <- c(
  "Resting" = "#2E86AB",
  "Swimming" = "#F18F01", 
  "Nursing" = "#36B155",      
  "High_activity" = "#E63946"
)

# PANEL 1: All behaviors together
plot1 <- ggplot(plot_data, aes(x = hour, y = predicted, color = behavior, fill = behavior)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.3, color = NA) +
  geom_line(size = 1.5) +
  scale_color_manual(values = behavior_colors) +
  scale_fill_manual(values = behavior_colors) +
  labs(x = "Hour of Day", y = "Probability")+ 
       #title = "A) Population-Level Activity Budget") +
  theme_classic() +
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    legend.position = "top"
  ) +
  scale_x_continuous(breaks = seq(0, 24, by = 4)) +
  ylim(0, max(plot_data$upper) * 1.05) +
  guides(color = guide_legend(title = NULL, override.aes = list(linewidth = 2)),
         fill = guide_legend(title = NULL))

# PANEL 2: Nursing only (zoomed in)
nursing_data <- plot_data[plot_data$behavior == "Nursing", ]

# Find peak for annotation
peak_hour <- nursing_data$hour[which.max(nursing_data$predicted)]
peak_prob <- max(nursing_data$predicted)
min_hour <- nursing_data$hour[which.min(nursing_data$predicted)]
min_prob <- min(nursing_data$predicted)

plot2 <- ggplot(nursing_data, aes(x = hour, y = predicted)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = behavior_colors["Nursing"], alpha = 0.3) +
  geom_line(color = behavior_colors["Nursing"], size = 1.5) +
  #geom_point(aes(x = peak_hour, y = peak_prob), color = "darkred", size = 4) +
  labs(x = "Hour of Day", 
       y = "Probability")+ 
       #title = "B) ") +
  theme_classic() +
  theme(
    plot.title = element_text(size = 18, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14)
  ) +
  scale_x_continuous(breaks = seq(0, 24, by = 4)) +
  ylim(0, max(nursing_data$upper) * 1.2)

# COMBINE PANELS
combined_plot <- grid.arrange(plot1, plot2, ncol = 2, widths = c(1, 1))

# SAVE THE COMBINED PLOT
ggsave("two_panel_activity_budget.png", combined_plot, 
       width = 16, height = 8, dpi = 300, bg = "white")

# SAVE PLOT DATA FOR LATER USE
save(plot_data, behavior_colors, file = "plot_data_for_reuse.RData")
cat("\n=== PLOT DATA SAVED TO 'plot_data_for_reuse.RData' ===\n")

cat("\n=== TWO-PANEL PLOT SAVED AS 'two_panel_activity_budget.png' ===\n")

# PRINT SUMMARY STATS
cat("\n### NURSING BEHAVIOR STATISTICS ###\n")
cat(paste("Peak time:", sprintf("%.1f", peak_hour), "hours\n"))
cat(paste("Peak probability:", sprintf("%.1f%%", peak_prob*100), "\n"))
cat(paste("Minimum time:", sprintf("%.1f", min_hour), "hours\n"))
cat(paste("Minimum probability:", sprintf("%.1f%%", min_prob*100), "\n"))
cat(paste("Daily range:", sprintf("%.1f%%", (peak_prob-min_prob)*100), "\n"))

cat("\n=== ANALYSIS COMPLETE ===\n")

###################################################### 
# continue here if not reloading gam models and data
# POPULATION-LEVEL PREDICTIONS
if (length(model_list) > 0) {
  plot_data <- data.frame()
  
  # Get unique pup IDs for predictions
  pup_ids <- unique(output_expanded$pup_id)
  
  # Prediction grid
  new_hours <- data.frame(
    hour = seq(0, 23.99, length.out = 200),
    hour_cos = cos(2 * pi * seq(0, 23.99, length.out = 200) / 24),
    hour_sin = sin(2 * pi * seq(0, 23.99, length.out = 200) / 24),
    pup_id = pup_ids[1]  # Use first pup as dummy
  )
  
  for (b in names(model_list)) {
    mod <- model_list[[b]]
    
    # POPULATION-LEVEL PREDICTION (exclude random effects)
    preds <- predict(mod, newdata = new_hours, se.fit = TRUE, exclude = "s(pup_id)")
    
    # Convert to probability scale
    predicted_prob <- plogis(preds$fit)
    lower_prob <- plogis(preds$fit - 1.96 * preds$se.fit)
    upper_prob <- plogis(preds$fit + 1.96 * preds$se.fit)
    
    plot_data <- rbind(plot_data, data.frame(
      hour = new_hours$hour,
      predicted = predicted_prob,
      lower = lower_prob,
      upper = upper_prob,
      behavior = b
    ))
  }
  
  # Define colors with nursing highlighted
  behavior_colors <- c(
    "Resting" = "#2E86AB",
    "Swimming" = "#F18F01", 
    "Nursing" = "#36B155",      
    "High_activity" = "#E63946"
  )
  
  # PANEL 1: All behaviors together
  plot1 <- ggplot(plot_data, aes(x = hour, y = predicted, color = behavior, fill = behavior)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.3, color = NA) +
    geom_line(size = 1.5) +
    scale_color_manual(values = behavior_colors) +
    scale_fill_manual(values = behavior_colors) +
    labs(x = "Hour of Day", y = "Probability", 
         title = "A) Population-Level Activity Budget") +
    theme_classic() +
    theme(
      plot.title = element_text(size = 18, face = "bold"),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 16),
      legend.position = "right"
    ) +
    scale_x_continuous(breaks = seq(0, 24, by = 4)) +
    ylim(0, max(plot_data$upper) * 1.05) +
    guides(color = guide_legend(title = "Behavior", override.aes = list(size = 2)),
           fill = guide_legend(title = "Behavior"))
  
  # PANEL 2: Nursing only (zoomed in)
  nursing_data <- plot_data[plot_data$behavior == "Nursing", ]
  
  # Find peak for annotation
  peak_hour <- nursing_data$hour[which.max(nursing_data$predicted)]
  peak_prob <- max(nursing_data$predicted)
  min_hour <- nursing_data$hour[which.min(nursing_data$predicted)]
  min_prob <- min(nursing_data$predicted)
  
  plot2 <- ggplot(nursing_data, aes(x = hour, y = predicted)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), fill = behavior_colors["Nursing"], alpha = 0.3) +
    geom_line(color = behavior_colors["Nursing"], size = 1.5) +
    geom_point(aes(x = peak_hour, y = peak_prob), color = "darkred", size = 4) +
    geom_text(aes(x = peak_hour, y = peak_prob) + 
              #label = paste0("Peak: ", sprintf("%.1f", peak_hour), "h\n", 
              #              sprintf("%.1f%%", peak_prob*100)),
              #vjust = -0.5, hjust = 0.5, color = "darkred", size = 5, fontface = "bold") +
    #geom_point(aes(x = min_hour, y = min_prob), color = "darkgreen", size = 4) +
    #geom_text(aes(x = min_hour, y = min_prob)+ 
              #label = paste0("Min: ", sprintf("%.1f", min_hour), "h\n", 
               #             sprintf("%.1f%%", min_prob*100)),
              #vjust = 1.2, hjust = 0.5, color = "darkgreen", size = 5, fontface = "bold") +
    labs(x = "Hour of Day", 
         y = "Probability", 
         title = "B) Nursing Behavior (Detail)") +
    theme_classic() +
    theme(
      plot.title = element_text(size = 18, face = "bold"),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14)
    ) +
    scale_x_continuous(breaks = seq(0, 24, by = 4)) +
    ylim(0, max(nursing_data$upper) * 1.2)
  
  # COMBINE PANELS
  combined_plot <- grid.arrange(plot1, plot2, ncol = 2, widths = c(1, 1))
  
  # SAVE THE COMBINED PLOT
  ggsave("two_panel_activity_budget.png", combined_plot, 
         width = 16, height = 8, dpi = 300, bg = "white")
  
  # SAVE PLOT DATA FOR LATER USE
  save(plot_data, behavior_colors, file = "plot_data_for_reuse.RData")
  cat("\n=== PLOT DATA SAVED TO 'plot_data_for_reuse.RData' ===\n")
  
  cat("\n=== TWO-PANEL PLOT SAVED AS 'two_panel_activity_budget.png' ===\n")
  
  # PRINT SUMMARY STATS
  cat("\n### NURSING BEHAVIOR STATISTICS ###\n")
  cat(paste("Peak time:", sprintf("%.1f", peak_hour), "hours\n"))
  cat(paste("Peak probability:", sprintf("%.1f%%", peak_prob*100), "\n"))
  cat(paste("Minimum time:", sprintf("%.1f", min_hour), "hours\n"))
  cat(paste("Minimum probability:", sprintf("%.1f%%", min_prob*100), "\n"))
  cat(paste("Daily range:", sprintf("%.1f%%", (peak_prob-min_prob)*100), "\n"))
  
} else {
  cat("No models were successfully fitted.\n")
}

cat("\n=== ANALYSIS COMPLETE ===\n")

# INSTRUCTIONS FOR REUSING SAVED DATA:
cat("\n=== TO REUSE SAVED DATA (without rerunning models) ===\n")
cat("1. Load models: load('gam_models_and_data.RData')\n")
cat("2. Load plot data: load('plot_data_for_reuse.RData')\n")
cat("3. Then you can modify and recreate plots without refitting models\n")

############################################################################################
# including autoregression and random splines not intercepts
############################################################################################

# library(mgcv)
# library(ggplot2)
# library(dplyr)
# library(tidyr)
# library(lubridate)
# library(readr)
# library(knitr)

# # Load and preprocess
# output <- read_csv("allpups_predictions_final.csv") 
# output$Timestamp <- mdy_hms(output$Timestamp)
# output$hour <- as.numeric(format(output$Timestamp, "%H")) + as.numeric(format(output$Timestamp, "%M")) / 60

# # Create time ordering
# output <- output %>%
#   arrange(Tag.ID, Timestamp) %>%
#   group_by(Tag.ID) %>%
#   mutate(time_order = row_number()) %>%
#   ungroup()

# # Map behaviors
# behaviors <- c(
#   "-1" = "Uncertain",
#   "0" = "Resting", 
#   "1" = "Nursing",
#   "2" = "High_activity",
#   "3" = "Inactive",
#   "4" = "Swimming"
# )
# output$behavior_label <- behaviors[as.character(output$Filtered_Predicted_Behavior)]

# # Create binary indicators and lagged variables for autoregression
# output_expanded <- output %>%
#   filter(!is.na(behavior_label)) %>%
#   select(Tag.ID, hour, time_order, behavior_label, Timestamp) %>%
#   mutate(
#     is_resting = as.numeric(behavior_label == "Resting"),
#     is_nursing = as.numeric(behavior_label == "Nursing"), 
#     is_high_activity = as.numeric(behavior_label == "High_activity"),
#     is_swimming = as.numeric(behavior_label == "Swimming"),
#     pup_id = as.factor(Tag.ID),
#     hour_cos = cos(2 * pi * hour / 24),
#     hour_sin = sin(2 * pi * hour / 24)
#   ) %>%
#   # Add lagged variables for autoregression (within each pup)
#   group_by(pup_id) %>%
#   mutate(
#     lag1_resting = lag(is_resting, 1),
#     lag1_nursing = lag(is_nursing, 1),
#     lag1_high_activity = lag(is_high_activity, 1),
#     lag1_swimming = lag(is_swimming, 1),
#     # You can add multiple lags if needed
#     lag2_resting = lag(is_resting, 2),
#     lag2_nursing = lag(is_nursing, 2),
#     lag2_high_activity = lag(is_high_activity, 2),
#     lag2_swimming = lag(is_swimming, 2)
#   ) %>%
#   ungroup() %>%
#   # Remove rows with missing lagged values
#   filter(!is.na(lag1_resting))

# # THIN PLATE CIRCULAR GAMM MODELS WITH RANDOM SPLINES AND AUTOREGRESSION
# model_list <- list()
# selected_behaviors <- c("Resting", "Swimming", "Nursing", "High_activity")
# behavior_columns <- c("is_resting", "is_swimming", "is_nursing", "is_high_activity")
# lag_columns <- c("lag1_resting", "lag1_swimming", "lag1_nursing", "lag1_high_activity")
# names(behavior_columns) <- selected_behaviors
# names(lag_columns) <- selected_behaviors

# cat("=== FITTING THIN PLATE CIRCULAR MODELS WITH RANDOM SPLINES AND AUTOREGRESSION ===\n")

# for (i in seq_along(selected_behaviors)) {
#   b <- selected_behaviors[i]
#   col_name <- behavior_columns[i]
#   lag_name <- lag_columns[i]
  
#   behavior_count <- sum(output_expanded[[col_name]], na.rm = TRUE)
#   cat(paste("\n--- Processing", b, "behavior ---\n"))
#   cat(paste("Total occurrences:", behavior_count, "\n"))
  
#   if (behavior_count < 100) {
#     cat(paste("Skipping", b, "- insufficient data\n"))
#     next
#   }
  
#   # Fit thin plate circular GAM with random splines and autoregression
#   cat("Fitting thin plate circular GAMM with random splines and autoregression...\n")
  
#   # Option 1: Random splines for each pup + autoregression
#   model_formula <- paste(col_name, "~ s(hour_cos, hour_sin, bs = 'tp', k = 8) +", 
#                         "s(hour_cos, hour_sin, by = pup_id, bs = 'tp', k = 6) +",
#                         "s(pup_id, bs = 're') +",
#                         lag_name)
  
#   m_tp <- tryCatch({
#     gam(as.formula(model_formula),
#         data = output_expanded,
#         family = binomial,
#         method = "REML")
#   }, error = function(e) {
#     cat("Full model failed:", e$message, "\n")
    
#     # Fallback: Try simpler random spline model
#     cat("Trying simpler random spline model...\n")
#     simple_formula <- paste(col_name, "~ s(hour_cos, hour_sin, bs = 'tp', k = 8) +", 
#                            "s(hour_cos, hour_sin, pup_id, bs = 'fs', k = 5) +",
#                            lag_name)
    
#     tryCatch({
#       gam(as.formula(simple_formula),
#           data = output_expanded,
#           family = binomial,
#           method = "REML")
#     }, error = function(e2) {
#       cat("Simplified model also failed:", e2$message, "\n")
#       return(NULL)
#     })
#   })
  
#   if (!is.null(m_tp)) {
#     model_list[[b]] <- m_tp
#     cat("Model fitted successfully\n")
    
#     # Quick model summary
#     cat("AIC:", AIC(m_tp), "\n")
#     cat("Deviance explained:", round(summary(m_tp)$dev.expl * 100, 1), "%\n")
    
#     # Check for autoregression effect
#     model_summary <- summary(m_tp)
#     if (lag_name %in% rownames(model_summary$p.table)) {
#       lag_coef <- model_summary$p.table[lag_name, "Estimate"]
#       lag_pval <- model_summary$p.table[lag_name, "Pr(>|t|)"]
#       cat(paste("Autoregression coefficient:", round(lag_coef, 3), 
#                 "p-value:", round(lag_pval, 4), "\n"))
#     }
#   }
# }

# # POPULATION-LEVEL PREDICTIONS (Modified for random splines)
# if (length(model_list) > 0) {
#   plot_data <- data.frame()
#   individual_data <- data.frame()
  
#   # Get unique pup IDs for predictions
#   pup_ids <- unique(output_expanded$pup_id)
  
#   # Prediction grid for population-level (average across individuals)
#   new_hours <- data.frame(
#     hour = seq(0, 23.99, length.out = 200),
#     hour_cos = cos(2 * pi * seq(0, 23.99, length.out = 200) / 24),
#     hour_sin = sin(2 * pi * seq(0, 23.99, length.out = 200) / 24)
#   )
  
#   for (b in names(model_list)) {
#     mod <- model_list[[b]]
#     lag_name <- lag_columns[b]
    
#     # For population predictions, we need to handle the lagged term
#     # Set lagged variable to the overall mean for population-level predictions
#     overall_mean_lag <- mean(output_expanded[[lag_name]], na.rm = TRUE)
    
#     pred_data_pop <- new_hours
#     pred_data_pop[[lag_name]] <- overall_mean_lag
#     pred_data_pop$pup_id <- pup_ids[1]  # Dummy pup ID
    
#     # POPULATION-LEVEL PREDICTION (exclude individual-specific random effects)
#     # This will depend on your exact model structure
#     preds <- tryCatch({
#       # Try to exclude random spline terms
#       if ("s(hour_cos,hour_sin):pup_id" %in% names(mod$smooth)) {
#         predict(mod, newdata = pred_data_pop, se.fit = TRUE, 
#                 exclude = c("s(pup_id)", "s(hour_cos,hour_sin):pup_id"))
#       } else {
#         predict(mod, newdata = pred_data_pop, se.fit = TRUE, 
#                 exclude = "s(pup_id)")
#       }
#     }, error = function(e) {
#       cat("Prediction error for", b, ":", e$message, "\n")
#       predict(mod, newdata = pred_data_pop, se.fit = TRUE)
#     })
    
#     # Convert to probability scale
#     predicted_prob <- plogis(preds$fit)
#     lower_prob <- plogis(preds$fit - 1.96 * preds$se.fit)
#     upper_prob <- plogis(preds$fit + 1.96 * preds$se.fit)
    
#     plot_data <- rbind(plot_data, data.frame(
#       hour = new_hours$hour,
#       predicted = predicted_prob,
#       lower = lower_prob,
#       upper = upper_prob,
#       behavior = b
#     ))
    
#     # INDIVIDUAL PUP PREDICTIONS (first 5 pups for comparison)
#     for (pup in pup_ids[1:min(5, length(pup_ids))]) {
#       pred_data_ind <- new_hours
#       pred_data_ind$pup_id <- pup
#       pred_data_ind[[lag_name]] <- overall_mean_lag
      
#       preds_ind <- predict(mod, newdata = pred_data_ind)
#       predicted_prob_ind <- plogis(preds_ind)
      
#       individual_data <- rbind(individual_data, data.frame(
#         hour = new_hours$hour,
#         predicted = predicted_prob_ind,
#         behavior = b,
#         pup_id = as.character(pup)
#       ))
#     }
#   }
  
#   # MAIN POPULATION-LEVEL PLOT
#   plot1 <- ggplot(plot_data, aes(x = hour, y = predicted, color = behavior, fill = behavior)) +
#     geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
#     geom_line(size = 1.2) +
#     labs(x = "Hour of Day", y = "Probability", 
#          title = "Population-Level Activity Budget",
#          subtitle = "Thin plate circular GAM with random splines and autoregression") +
#     theme_classic() +
#     theme(legend.position = "bottom") +
#     scale_x_continuous(breaks = seq(0, 24, by = 2)) +
#     ylim(0, 1)
  
#   # FACETED PLOT
#   plot2 <- ggplot(plot_data, aes(x = hour, y = predicted)) +
#     geom_ribbon(aes(ymin = lower, ymax = upper), fill = "steelblue", alpha = 0.3) +
#     geom_line(color = "steelblue", size = 1.2) +
#     facet_wrap(~ behavior, scales = "free_y") +
#     labs(x = "Hour of Day", y = "Probability", 
#          title = "Population-Level Activity Budget by Behavior",
#          subtitle = "Thin plate circular GAM with random splines and autoregression") +
#     theme_classic() +
#     scale_x_continuous(breaks = seq(0, 24, by = 6))
  
#   # INDIVIDUAL VS POPULATION COMPARISON
#   plot3 <- ggplot() +
#     geom_line(data = individual_data, aes(x = hour, y = predicted, group = pup_id), 
#               color = "gray70", alpha = 0.7, size = 0.8) +
#     geom_ribbon(data = plot_data, aes(x = hour, ymin = lower, ymax = upper), 
#                 fill = "red", alpha = 0.3) +
#     geom_line(data = plot_data, aes(x = hour, y = predicted), 
#               color = "red", size = 1.5) +
#     facet_wrap(~ behavior, scales = "free_y") +
#     labs(x = "Hour of Day", y = "Probability", 
#          title = "Population vs Individual Patterns",
#          subtitle = "Gray lines = individual pups, Red = population average with 95% CI\n(Random splines + autoregression)") +
#     theme_classic() +
#     scale_x_continuous(breaks = seq(0, 24, by = 6))
  
# # ENHANCED INDIVIDUAL PLOTS WITH ANNOTATIONS
#   for (behav in unique(plot_data$behavior)) {
#     behav_data <- plot_data[plot_data$behavior == behav, ]
    
#     # Find peak and minimum
#     peak_hour <- behav_data$hour[which.max(behav_data$predicted)]
#     peak_prob <- max(behav_data$predicted)
#     min_hour <- behav_data$hour[which.min(behav_data$predicted)]
#     min_prob <- min(behav_data$predicted)
    
#     # Create enhanced plot
#     p_enhanced <- ggplot(behav_data, aes(x = hour, y = predicted)) +
#       geom_ribbon(aes(ymin = lower, ymax = upper), fill = "steelblue", alpha = 0.3) +
#       geom_line(color = "steelblue", size = 1.5) +
#       geom_point(aes(x = peak_hour, y = peak_prob), color = "red", size = 3) +
#       geom_text(aes(x = peak_hour, y = peak_prob), 
#                 label = paste0("Peak: ", sprintf("%.1f", peak_hour), "h\n", 
#                               sprintf("%.1f%%", peak_prob*100)),
#                 vjust = -0.5, hjust = 0.5, color = "red", size = 3) +
#       geom_point(aes(x = min_hour, y = min_prob), color = "darkgreen", size = 3) +
#       geom_text(aes(x = min_hour, y = min_prob), 
#                 label = paste0("Min: ", sprintf("%.1f", min_hour), "h\n", 
#                               sprintf("%.1f%%", min_prob*100)),
#                 vjust = 1.2, hjust = 0.5, color = "darkgreen", size = 3) +
#       labs(x = "Hour of Day", 
#            y = "Probability", 
#            title = paste("Population-Level", behav, "Behavior Pattern"),
#            subtitle = paste0("Thin plate circular GAM - Peak: ", sprintf("%.1f", peak_hour), 
#                            "h, Range: ", sprintf("%.1f%%", (peak_prob-min_prob)*100))) +
#       theme_classic() +
#       theme(
#         plot.title = element_text(size = 16, face = "bold"),
#         plot.subtitle = element_text(size = 11),
#         axis.title = element_text(size = 12),
#         axis.text = element_text(size = 10)
#       ) +
#       scale_x_continuous(breaks = seq(0, 24, by = 2)) +
#       ylim(0, max(behav_data$upper) * 1.15)
    
#     # Save individual behavior plot
#     filename <- paste0("thinplate_", tolower(gsub("_", "", behav)), "_behavior.png")
#     ggsave(filename, plot = p_enhanced, width = 10, height = 6, dpi = 300)
#     cat(paste("Saved:", filename, "\n"))
#   }
  
#   # ENHANCED MODEL DIAGNOSTICS
#   if (length(model_list) > 0) {
#     for (behav_name in names(model_list)) {
#       model <- model_list[[behav_name]]
      
#       cat(paste("\n### MODEL DIAGNOSTICS FOR", behav_name, "###\n"))
#       cat("Model terms:\n")
#       print(summary(model)$s.table)
#       cat("\nParametric coefficients (including autoregression):\n")
#       print(summary(model)$p.table)
      
#       # Check residual autocorrelation
#       cat("\nChecking residual autocorrelation...\n")
#       residuals <- residuals(model)
      
#       # Simple autocorrelation check (you might want to do this by individual)
#       if (length(residuals) > 1) {
#         acf_result <- acf(residuals, plot = FALSE, lag.max = 5)
#         cat("Residual autocorrelations (lags 1-5):", round(acf_result$acf[2:6], 3), "\n")
#       }
#     }
#   }
  
#   # Save plots with updated names
#   ggsave("randomspline_autoregress_activity_budget.png", plot = plot1, width = 12, height = 8, dpi = 300)
#   ggsave("randomspline_autoregress_behaviors_faceted.png", plot = plot2, width = 12, height = 8, dpi = 300)
#   ggsave("randomspline_autoregress_population_vs_individual.png", plot = plot3, width = 12, height = 8, dpi = 300)
  
# } else {
#   cat("No models were successfully fitted.\n")
# }

# cat("\n=== RANDOM SPLINE + AUTOREGRESSION GAM ANALYSIS COMPLETE ===\n")


############ above is only thin plate - below compares against cyclic cubic

# library(mgcv)
# library(nlme)
# library(ggplot2)
# library(dplyr)
# library(tidyr)
# library(lubridate)
# library(readr)
# library(AICcmodavg)
# library(knitr)

# # Load and preprocess
# output <- read_csv("allpups_predictions_final.csv") 
# output$Timestamp <- mdy_hms(output$Timestamp)
# output$hour <- as.numeric(format(output$Timestamp, "%H")) + as.numeric(format(output$Timestamp, "%M")) / 60

# # Create time ordering for autoregression
# output <- output %>%
#   arrange(Tag.ID, Timestamp) %>%
#   group_by(Tag.ID) %>%
#   mutate(time_order = row_number()) %>%
#   ungroup()

# # Map behaviors
# behaviors <- c(
#   "-1" = "Uncertain",
#   "0" = "Resting", 
#   "1" = "Nursing",
#   "2" = "High_activity",
#   "3" = "Inactive",
#   "4" = "Swimming"
# )
# output$behavior_label <- behaviors[as.character(output$Filtered_Predicted_Behavior)]

# # Create binary indicators
# output_expanded <- output %>%
#   filter(!is.na(behavior_label)) %>%
#   select(Tag.ID, hour, time_order, behavior_label, Timestamp) %>%
#   mutate(
#     is_resting = as.numeric(behavior_label == "Resting"),
#     is_nursing = as.numeric(behavior_label == "Nursing"), 
#     is_high_activity = as.numeric(behavior_label == "High_activity"),
#     is_swimming = as.numeric(behavior_label == "Swimming"),
#     pup_id = as.factor(Tag.ID),
#     hour_cos = cos(2 * pi * hour / 24),
#     hour_sin = sin(2 * pi * hour / 24)
#   )

# # POPULATION-LEVEL GAMM MODELS WITH RANDOM EFFECTS
# model_list <- list()
# aicc_table <- data.frame()
# selected_behaviors <- c("Resting", "Swimming", "Nursing", "High_activity")
# behavior_columns <- c("is_resting", "is_swimming", "is_nursing", "is_high_activity")
# names(behavior_columns) <- selected_behaviors

# cat("=== FITTING POPULATION-LEVEL MODELS ===\n")

# for (i in seq_along(selected_behaviors)) {
#   b <- selected_behaviors[i]
#   col_name <- behavior_columns[i]
  
#   behavior_count <- sum(output_expanded[[col_name]], na.rm = TRUE)
#   cat(paste("\n--- Processing", b, "behavior ---\n"))
#   cat(paste("Total occurrences:", behavior_count, "\n"))
  
#   if (behavior_count < 100) {
#     cat(paste("Skipping", b, "- insufficient data\n"))
#     next
#   }
  
#   # Fit GAMM with random intercepts for pups
#   cat("Fitting population-level GAMM...\n")
  
#   # Try cyclic cubic spline first (most appropriate for circadian data)
#   m_cc <- tryCatch({
#     gam(as.formula(paste(col_name, "~ s(hour, bs = 'cc', k = 8) + s(pup_id, bs = 're')")),
#         data = output_expanded, 
#         family = binomial, 
#         method = "REML",
#         knots = list(hour = c(0, 24)))
#   }, error = function(e) {
#     cat("Cyclic cubic model failed:", e$message, "\n")
#     return(NULL)
#   })
  
#   # Try thin plate on circular coordinates as alternative
#   m_tp <- tryCatch({
#     gam(as.formula(paste(col_name, "~ s(hour_cos, hour_sin, bs = 'tp', k = 8) + s(pup_id, bs = 're')")),
#         data = output_expanded,
#         family = binomial,
#         method = "REML")
#   }, error = function(e) {
#     cat("Thin plate circular model failed:", e$message, "\n")
#     return(NULL)
#   })
  
#   # Store successful models
#   if (!is.null(m_cc)) {
#     model_list[[paste(b, "cc")]] <- m_cc
#     aicc_table <- rbind(aicc_table, data.frame(
#       behavior = b,
#       spline_type = "cc",
#       AIC = AIC(m_cc)
#     ))
#     cat("Cyclic cubic model fitted successfully\n")
#   }
  
#   if (!is.null(m_tp)) {
#     model_list[[paste(b, "tp")]] <- m_tp
#     aicc_table <- rbind(aicc_table, data.frame(
#       behavior = b,
#       spline_type = "tp", 
#       AIC = AIC(m_tp)
#     ))
#     cat("Thin plate circular model fitted successfully\n")
#   }
# }

# # Select best models
# if (nrow(aicc_table) > 0) {
#   best_models <- aicc_table %>%
#     group_by(behavior) %>%
#     slice_min(order_by = AIC, n = 1) %>%
#     ungroup()
  
#   cat("\n### Best Models Selected:\n")
#   print(kable(best_models, digits = 2))
  
#   # POPULATION-LEVEL PREDICTIONS
#   plot_data <- data.frame()
#   individual_data <- data.frame()
  
#   # Prediction grid
#   new_hours <- data.frame(
#     hour = seq(0, 23.99, length.out = 200),
#     hour_cos = cos(2 * pi * seq(0, 23.99, length.out = 200) / 24),
#     hour_sin = sin(2 * pi * seq(0, 23.99, length.out = 200) / 24)
#   )
  
#   # Get unique pup IDs
#   pup_ids <- unique(output_expanded$pup_id)
  
#   for (i in 1:nrow(best_models)) {
#     b <- best_models$behavior[i]
#     spline_type <- best_models$spline_type[i]
#     model_key <- paste(b, spline_type)
#     mod <- model_list[[model_key]]
    
#     if (is.null(mod)) next
    
#     # POPULATION-LEVEL PREDICTION (marginal effects, averaging over random effects)
#     pred_data <- new_hours
    
#     # Predict at population level (exclude random effects)
#     preds <- predict(mod, newdata = pred_data, se.fit = TRUE, exclude = "s(pup_id)")
    
#     # Convert to probability scale
#     predicted_prob <- plogis(preds$fit)
#     lower_prob <- plogis(preds$fit - 1.96 * preds$se.fit)
#     upper_prob <- plogis(preds$fit + 1.96 * preds$se.fit)
    
#     plot_data <- rbind(plot_data, data.frame(
#       hour = new_hours$hour,
#       predicted = predicted_prob,
#       lower = lower_prob,
#       upper = upper_prob,
#       behavior = b,
#       spline_type = spline_type
#     ))
    
#     # INDIVIDUAL PUP PREDICTIONS (for comparison/validation)
#     for (pup in pup_ids[1:min(5, length(pup_ids))]) {  # Show first 5 pups
#       pred_data_ind <- new_hours
#       pred_data_ind$pup_id <- pup
      
#       preds_ind <- predict(mod, newdata = pred_data_ind)
#       predicted_prob_ind <- plogis(preds_ind)
      
#       individual_data <- rbind(individual_data, data.frame(
#         hour = new_hours$hour,
#         predicted = predicted_prob_ind,
#         behavior = b,
#         pup_id = as.character(pup)
#       ))
#     }
#   }
  
#   # POPULATION-LEVEL ACTIVITY BUDGET PLOT
#   plot1 <- ggplot(plot_data, aes(x = hour, y = predicted, color = behavior, fill = behavior)) +
#     geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
#     geom_line(size = 1.2) +
#     labs(x = "Hour of Day", y = "Probability", 
#          title = "Population-Level Activity Budget",
#          subtitle = "GAM with random effects for individual pups, population-level curves shown") +
#     theme_classic() +
#     theme(legend.position = "bottom") +
#     scale_x_continuous(breaks = seq(0, 24, by = 2)) +
#     ylim(0, 1)
  
#   # FACETED POPULATION-LEVEL PLOT
#   plot2 <- ggplot(plot_data, aes(x = hour, y = predicted)) +
#     geom_ribbon(aes(ymin = lower, ymax = upper), fill = "steelblue", alpha = 0.3) +
#     geom_line(color = "steelblue", size = 1.2) +
#     facet_wrap(~ behavior, scales = "free_y") +
#     labs(x = "Hour of Day", y = "Probability", 
#          title = "Population-Level Activity Budget by Behavior",
#          subtitle = "Population-level curves from GAMMs with individual random effects") +
#     theme_classic() +
#     scale_x_continuous(breaks = seq(0, 24, by = 6))
  
#   # INDIVIDUAL VS POPULATION COMPARISON
#   plot3 <- ggplot() +
#     geom_line(data = individual_data, aes(x = hour, y = predicted, group = pup_id), 
#               color = "gray70", alpha = 0.7, size = 0.8) +
#     geom_ribbon(data = plot_data, aes(x = hour, ymin = lower, ymax = upper), 
#                 fill = "red", alpha = 0.3) +
#     geom_line(data = plot_data, aes(x = hour, y = predicted), 
#               color = "red", size = 1.5) +
#     facet_wrap(~ behavior, scales = "free_y") +
#     labs(x = "Hour of Day", y = "Probability", 
#          title = "Population vs Individual Patterns",
#          subtitle = "Gray lines = individual pups, Red = population average with 95% CI") +
#     theme_classic() +
#     scale_x_continuous(breaks = seq(0, 24, by = 6))
  
#   # ACTIVITY BUDGET TABLE (proportion of time in each behavior)
#   activity_budget <- plot_data %>%
#     group_by(behavior) %>%
#     summarise(
#       mean_prob = mean(predicted),
#       min_prob = min(predicted), 
#       max_prob = max(predicted),
#       .groups = "drop"
#     ) %>%
#     arrange(desc(mean_prob))
  
#   cat("\n### POPULATION-LEVEL ACTIVITY BUDGET SUMMARY ###\n")
#   print(kable(activity_budget, 
#               col.names = c("Behavior", "Mean Proportion", "Min", "Max"),
#               digits = 3))
  
#   # INDIVIDUAL BEHAVIOR PLOTS
#   individual_plots <- list()
  
#   for (behav in unique(plot_data$behavior)) {
#     behav_data <- plot_data[plot_data$behavior == behav, ]
    
#     # Create individual plot for this behavior
#     p <- ggplot(behav_data, aes(x = hour, y = predicted)) +
#       geom_ribbon(aes(ymin = lower, ymax = upper), fill = "steelblue", alpha = 0.3) +
#       geom_line(color = "steelblue", size = 1.5) +
#       labs(x = "Hour of Day", 
#            y = "Probability", 
#            title = paste("Population-Level", behav, "Behavior"),
#            subtitle = "GAM with random effects - population average with 95% CI") +
#       theme_classic() +
#       theme(
#         plot.title = element_text(size = 16, face = "bold"),
#         plot.subtitle = element_text(size = 12),
#         axis.title = element_text(size = 12),
#         axis.text = element_text(size = 10)
#       ) +
#       scale_x_continuous(breaks = seq(0, 24, by = 2)) +
#       ylim(0, max(behav_data$upper) * 1.05)
    
#     individual_plots[[behav]] <- p
    
#     # Save individual plot
#     filename <- paste0("population_", tolower(gsub("_", "", behav)), "_behavior.png")
#     ggsave(filename, plot = p, width = 10, height = 6, dpi = 300)
#     cat(paste("Saved:", filename, "\n"))
#   }
  
#   # ENHANCED INDIVIDUAL PLOTS WITH STATISTICAL ANNOTATIONS
#   enhanced_plots <- list()
  
#   for (behav in unique(plot_data$behavior)) {
#     behav_data <- plot_data[plot_data$behavior == behav, ]
    
#     # Find peak activity time
#     peak_hour <- behav_data$hour[which.max(behav_data$predicted)]
#     peak_prob <- max(behav_data$predicted)
    
#     # Find minimum activity time  
#     min_hour <- behav_data$hour[which.min(behav_data$predicted)]
#     min_prob <- min(behav_data$predicted)
    
#     # Create enhanced plot
#     p_enhanced <- ggplot(behav_data, aes(x = hour, y = predicted)) +
#       geom_ribbon(aes(ymin = lower, ymax = upper), fill = "steelblue", alpha = 0.3) +
#       geom_line(color = "steelblue", size = 1.5) +
#       # Add peak point
#       geom_point(aes(x = peak_hour, y = peak_prob), color = "red", size = 3) +
#       geom_text(aes(x = peak_hour, y = peak_prob), 
#                 label = paste0("Peak: ", sprintf("%.2f", peak_hour), "h\n", 
#                               sprintf("%.1f%%", peak_prob*100)),
#                 vjust = -0.5, hjust = 0.5, color = "red", size = 3) +
#       # Add minimum point
#       geom_point(aes(x = min_hour, y = min_prob), color = "darkgreen", size = 3) +
#       geom_text(aes(x = min_hour, y = min_prob), 
#                 label = paste0("Min: ", sprintf("%.2f", min_hour), "h\n", 
#                               sprintf("%.1f%%", min_prob*100)),
#                 vjust = 1.2, hjust = 0.5, color = "darkgreen", size = 3) +
#       labs(x = "Hour of Day", 
#            y = "Probability", 
#            title = paste("Population-Level", behav, "Behavior Pattern"),
#            subtitle = paste0("Peak at ", sprintf("%.1f", peak_hour), "h (", 
#                            sprintf("%.1f%%", peak_prob*100), "), ",
#                            "Minimum at ", sprintf("%.1f", min_hour), "h (", 
#                            sprintf("%.1f%%", min_prob*100), ")")) +
#       theme_classic() +
#       theme(
#         plot.title = element_text(size = 16, face = "bold"),
#         plot.subtitle = element_text(size = 11),
#         axis.title = element_text(size = 12),
#         axis.text = element_text(size = 10)
#       ) +
#       scale_x_continuous(breaks = seq(0, 24, by = 2)) +
#       ylim(0, max(behav_data$upper) * 1.15)  # Extra space for annotations
    
#     enhanced_plots[[behav]] <- p_enhanced
    
#     # Save enhanced plot
#     filename_enhanced <- paste0("enhanced_", tolower(gsub("_", "", behav)), "_behavior.png")
#     ggsave(filename_enhanced, plot = p_enhanced, width = 10, height = 6, dpi = 300)
#     cat(paste("Saved:", filename_enhanced, "\n"))
#   }
  
#   # SUMMARY STATISTICS FOR EACH BEHAVIOR
#   behavior_stats <- plot_data %>%
#     group_by(behavior) %>%
#     summarise(
#       peak_hour = hour[which.max(predicted)],
#       peak_probability = max(predicted),
#       min_hour = hour[which.min(predicted)],
#       min_probability = min(predicted),
#       mean_probability = mean(predicted),
#       amplitude = max(predicted) - min(predicted),
#       .groups = "drop"
#     ) %>%
#     arrange(desc(mean_probability))
  
#   cat("\n### DETAILED BEHAVIOR STATISTICS ###\n")
#   print(kable(behavior_stats, 
#               col.names = c("Behavior", "Peak Hour", "Peak Prob", "Min Hour", 
#                            "Min Prob", "Mean Prob", "Amplitude"),
#               digits = 2))
  
#   # Save plots
#   ggsave("population_activity_budget.png", plot = plot1, width = 12, height = 8, dpi = 300)
#   ggsave("population_behaviors_faceted.png", plot = plot2, width = 12, height = 8, dpi = 300)
#   ggsave("population_vs_individual.png", plot = plot3, width = 12, height = 8, dpi = 300)
  
#   # Model diagnostics for best model
#   best_model <- model_list[[paste(best_models$behavior[1], best_models$spline_type[1])]]
#   if (!is.null(best_model)) {
#     cat("\n### MODEL DIAGNOSTICS FOR", best_models$behavior[1], "###\n")
#     par(mfrow = c(2, 2))
#     gam.check(best_model)
    
#     # Summary
#     cat("\nModel Summary:\n")
#     print(summary(best_model))
#   }
  
# } else {
#   cat("No models were successfully fitted.\n")
# }

# cat("\n=== ANALYSIS COMPLETE ===\n")
# cat("Population-level activity budget represents the average behavioral patterns")
# cat("across all individuals, accounting for individual variation through random effects.\n")

############ averaged 30 min bins ############

# # Read in your data
# output <- read_csv("RawBB_predictions_with_probs.csv")

# # Convert Timestamp to POSIXct if it's not already
# output$Timestamp <- ymd_hms(output$Timestamp)

# # Get the hour as a decimal
# output <- output %>%
#   mutate(HourDecimal = hour(Timestamp) + minute(Timestamp) / 60)

# # Bin time into 48 half-hour slots (0-0.5, 0.5-1.0, ..., 23.5-24.0)
# breaks <- seq(0, 24, by = 0.5)
# output <- output %>%
#   mutate(HourBin = cut(HourDecimal, breaks = breaks, include.lowest = TRUE, right = FALSE))

# # Define a mapping from numeric codes to behavior names
# behavior_labels <- c(
#   "-1" = "Uncertain",
#   "0" = "Sleeping",
#   "1" = "Nursing",
#   "2" = "Active",
#   "3" = "Inactive",
#   "4" = "Swimming"
# )

# # Relabel behavior values
# output <- output %>%
#   mutate(Behavior_Label = recode(as.character(Filtered_Predicted_Behavior), !!!behavior_labels))

# # Count and pivot
# behavior_hourly <- output %>%
#   count(HourBin, Behavior_Label) %>%
#   pivot_wider(names_from = Behavior_Label,
#               values_from = n,
#               values_fill = 0)

# # Convert counts to proportions
# behavior_hourly_prop <- behavior_hourly %>%
#   rowwise() %>%
#   mutate(Total = sum(c_across(-HourBin))) %>%
#   ungroup() %>%
#   mutate(across(-c(HourBin, Total), ~ .x / Total))

# # Extract hour midpoints
# behavior_hourly_prop <- behavior_hourly_prop %>%
#   mutate(HourMid = as.numeric(as.character(HourBin)))

# # Fix the HourMid calculation
# behavior_hourly_prop$HourMid <- sapply(behavior_hourly_prop$HourBin, function(bin) {
#   # Extract the lower bound of the interval
#   lower <- as.numeric(sub("\\[(.+),.*", "\\1", as.character(bin)))
#   # Add half the bin width (0.25 hours)
#   lower + 0.25
# })

# # Prepare long format data for modeling
# behavior_long <- behavior_hourly_prop %>%
#   select(-Total) %>%
#   pivot_longer(cols = -c(HourBin, HourMid), 
#                names_to = "Behavior", 
#                values_to = "Proportion")

# # Now fit GAM models for each behavior
# behaviors <- unique(behavior_long$Behavior)
# gam_models <- list()

# for (behavior in behaviors) {
#   # Subset data for this behavior
#   behavior_data <- behavior_long %>% filter(Behavior == behavior)
  
#   # 1. Cyclic P-spline GAM model
#   gam_models[[paste0(behavior, "_pspline")]] <- gam(
#     Proportion ~ s(HourMid, bs = "cp", k = 10),  # cp = cyclic P-spline
#     data = behavior_data,
#     method = "REML"
#   )
  
#   # 2. Cyclic cubic spline GAM model (alternative to thin plate for circular data)
#   gam_models[[paste0(behavior, "_cyclic")]] <- gam(
#     Proportion ~ s(HourMid, bs = "cc", k = 10),  # cc = cyclic cubic spline
#     data = behavior_data,
#     method = "REML"
#   )
  
#   # 3. Thin plate spline GAM model
#   #gam_models[[paste0(behavior, "_thinplate")]] <- gam(
#    # Proportion ~ s(HourMid, bs = "tp", k = 10),  # tp = thin plate spline
#    # data = behavior_data,
#    # method = "REML"
#   #)
  
#   # 4. Thin plate spline with cyclic constraint (using duplicate endpoints)
#   # Create augmented data with wrapping to handle circularity
#   behavior_data_augmented <- behavior_data
#   # Add data points at the beginning and end to encourage smooth cycling
#   start_points <- behavior_data %>% 
#     filter(HourMid <= 2) %>% 
#     mutate(HourMid = HourMid + 24)
#   end_points <- behavior_data %>% 
#     filter(HourMid >= 22) %>% 
#     mutate(HourMid = HourMid - 24)
  
#   behavior_data_augmented <- bind_rows(behavior_data_augmented, start_points, end_points)
  
#   gam_models[[paste0(behavior, "_thinplate_cyclic")]] <- gam(
#     Proportion ~ s(HourMid, bs = "tp", k = 10),  # tp = thin plate spline with more knots
#     data = behavior_data_augmented,
#     method = "REML"
#   )
# }

# # Visualize all models
# pdf("GAM_behavior_models_all.pdf", width = 16, height = 8)
# for (behavior in behaviors) {
#   par(mfrow = c(2, 2))
#   plot(gam_models[[paste0(behavior, "_pspline")]], 
#        main = paste0(behavior, " - Cyclic P-spline GAM"))
#   plot(gam_models[[paste0(behavior, "_cyclic")]], 
#        main = paste0(behavior, " - Cyclic cubic spline GAM"))
#   #plot(gam_models[[paste0(behavior, "_thinplate")]], 
#    #    main = paste0(behavior, " - Thin plate spline GAM"))
#   plot(gam_models[[paste0(behavior, "_thinplate_cyclic")]], 
#        main = paste0(behavior, " - Thin plate spline (cyclic) GAM"))
# }
# dev.off()

# # Generate predictions for plotting
# hour_seq <- seq(0, 24, length.out = 100)
# prediction_data <- data.frame(HourMid = hour_seq)

# # Create a dataframe to store all predictions
# all_predictions <- data.frame(Hour = hour_seq)

# for (behavior in behaviors) {
  
#   # Get predictions for P-spline model
#   pred_pspline <- predict(gam_models[[paste0(behavior, "_pspline")]], 
#                          newdata = prediction_data, 
#                          se.fit = TRUE)
  
#   # Get predictions for cyclic cubic spline model
#   pred_cyclic <- predict(gam_models[[paste0(behavior, "_cyclic")]], 
#                         newdata = prediction_data, 
#                         se.fit = TRUE)
  
#   # Get predictions for thin plate spline model
#   #pred_thinplate <- predict(gam_models[[paste0(behavior, "_thinplate")]], 
#                            #newdata = prediction_data, 
#                            #se.fit = TRUE)
  
#   # Get predictions for cyclic thin plate spline model
#   pred_thinplate_cyclic <- predict(gam_models[[paste0(behavior, "_thinplate_cyclic")]], 
#                                   newdata = prediction_data, 
#                                   se.fit = TRUE)
  
#   # Add to prediction dataframe
#   all_predictions[[paste0(behavior, "_pspline")]] <- pred_pspline$fit
#   all_predictions[[paste0(behavior, "_pspline_upper")]] <- pred_pspline$fit + 1.96 * pred_pspline$se.fit
#   all_predictions[[paste0(behavior, "_pspline_lower")]] <- pred_pspline$fit - 1.96 * pred_pspline$se.fit
  
#   all_predictions[[paste0(behavior, "_cyclic")]] <- pred_cyclic$fit
#   all_predictions[[paste0(behavior, "_cyclic_upper")]] <- pred_cyclic$fit + 1.96 * pred_cyclic$se.fit
#   all_predictions[[paste0(behavior, "_cyclic_lower")]] <- pred_cyclic$fit - 1.96 * pred_cyclic$se.fit
  
#   #all_predictions[[paste0(behavior, "_thinplate")]] <- pred_thinplate$fit
#   #all_predictions[[paste0(behavior, "_thinplate_upper")]] <- pred_thinplate$fit + 1.96 * pred_thinplate$se.fit
#   #all_predictions[[paste0(behavior, "_thinplate_lower")]] <- pred_thinplate$fit - 1.96 * pred_thinplate$se.fit
  
#   all_predictions[[paste0(behavior, "_thinplate_cyclic")]] <- pred_thinplate_cyclic$fit
#   all_predictions[[paste0(behavior, "_thinplate_cyclic_upper")]] <- pred_thinplate_cyclic$fit + 1.96 * pred_thinplate_cyclic$se.fit
#   all_predictions[[paste0(behavior, "_thinplate_cyclic_lower")]] <- pred_thinplate_cyclic$fit - 1.96 * pred_thinplate_cyclic$se.fit
# }

# # Create ggplot visualizations comparing all models
# for (behavior in behaviors) {
#   # Convert to long format for easier plotting
#   plot_data <- behavior_long %>% filter(Behavior == behavior)
  
#   # Create prediction data for this behavior
#   pred_data <- all_predictions %>%
#     select(Hour, 
#            starts_with(paste0(behavior, "_"))) %>%
#     rename(
#       P_Spline = paste0(behavior, "_pspline"),
#       P_Spline_Upper = paste0(behavior, "_pspline_upper"),
#       P_Spline_Lower = paste0(behavior, "_pspline_lower"),
#       Cyclic = paste0(behavior, "_cyclic"),
#       Cyclic_Upper = paste0(behavior, "_cyclic_upper"),
#       Cyclic_Lower = paste0(behavior, "_cyclic_lower"),
#       #Thin_Plate = paste0(behavior, "_thinplate"),
#       #Thin_Plate_Upper = paste0(behavior, "_thinplate_upper"),
#       #Thin_Plate_Lower = paste0(behavior, "_thinplate_lower"),
#       Thin_Plate_Cyclic = paste0(behavior, "_thinplate_cyclic"),
#       Thin_Plate_Cyclic_Upper = paste0(behavior, "_thinplate_cyclic_upper"),
#       Thin_Plate_Cyclic_Lower = paste0(behavior, "_thinplate_cyclic_lower")
#     ) %>%
#     pivot_longer(cols = -Hour,
#                  names_to = "Model", 
#                  values_to = "Proportion")
  
#   # Filter for main model fits (not confidence intervals)
#   main_models <- c("P_Spline", "Cyclic", "Thin_Plate_Cyclic")
  
#   # Basic plot comparing all three models
#   p <- ggplot() +
#     #geom_point(data = plot_data, 
#      #          aes(x = HourMid, y = Proportion),
#       #         alpha = 0.5, color = "black") +
#     geom_line(data = pred_data %>% filter(Model %in% main_models),
#               aes(x = Hour, y = Proportion, color = Model, linetype = Model),
#               linewidth = 2) +
#     scale_color_manual(values = c("P_Spline" = "red", 
#                                  "Cyclic" = "blue", 
#                                  "Thin_Plate_Cyclic" = "green")) +
#     scale_linetype_manual(values = c("P_Spline" = "solid", 
#                                    "Cyclic" = "dashed", 
#                                    "Thin_Plate_Cyclic" = "dotted")) +
#     theme_classic() +
#     theme(
#       legend.position = "right"
#     ) +
#     labs(title = paste("GAM Model Comparison for", behavior),
#          x = "Hour of Day",
#          y = "Proportion") +
#     scale_x_continuous(breaks = seq(0, 24, by = 3),
#                       limits = c(0, 24))
  
#   # Save the plot
#   ggsave(paste0("GAM_Comparison_", behavior, ".png"), p, width = 12, height = 8)
# }

# # Print model summaries and compare AIC values
# cat("Model Comparison Summary:\n")
# cat("========================\n\n")

# comparison_results <- data.frame(
#   Behavior = character(),
#   P_Spline_AIC = numeric(),
#   Cyclic_AIC = numeric(),
#   Thin_Plate_Cyclic_AIC = numeric(),
#   Best_Model = character(),
#   stringsAsFactors = FALSE
# )

# for (behavior in behaviors) {
#   cat("\n", behavior, "Model Comparison:\n")
#   cat("================================\n")
  
#   # Get AIC values
#   pspline_aic <- AIC(gam_models[[paste0(behavior, "_pspline")]])
#   cyclic_aic <- AIC(gam_models[[paste0(behavior, "_cyclic")]])
#   thinplate_cyclic_aic <- AIC(gam_models[[paste0(behavior, "_thinplate_cyclic")]])
  
#   # Find best model
#   aic_values <- c(pspline_aic, cyclic_aic, thinplate_cyclic_aic)
#   model_names <- c("P-Spline", "Cyclic", "Thin Plate Cyclic")
#   best_model <- model_names[which.min(aic_values)]
  
#   cat("P-Spline AIC:", round(pspline_aic, 3), "\n")
#   cat("Cyclic AIC:", round(cyclic_aic, 3), "\n")
#   cat("Thin Plate Cyclic AIC:", round(thinplate_cyclic_aic, 3), "\n")
#   cat("Best Model:", best_model, "\n")
  
#   # Add to comparison results
#   comparison_results <- rbind(comparison_results, data.frame(
#     Behavior = behavior,
#     P_Spline_AIC = pspline_aic,
#     Cyclic_AIC = cyclic_aic,
#     Thin_Plate_Cyclic_AIC = thinplate_cyclic_aic,
#     Best_Model = best_model
#   ))
  
#   cat("\nModel summaries:\n")
#   cat("P-spline Model Summary:\n")
#   print(summary(gam_models[[paste0(behavior, "_pspline")]]))
  
#   cat("\nCyclic Cubic Spline Model Summary:\n")
#   print(summary(gam_models[[paste0(behavior, "_cyclic")]]))
  
#   cat("\nThin Plate Spline (Cyclic) Model Summary:\n")
#   print(summary(gam_models[[paste0(behavior, "_thinplate_cyclic")]]))
# }

# # Print overall comparison table
# cat("\n\nOverall Model Comparison:\n")
# cat("=========================\n")
# print(comparison_results)

# # Save comparison results
# write.csv(comparison_results, "GAM_Model_Comparison.csv", row.names = FALSE)

# ############ combining behaviors to one plot - comparing spline types ############

# # Create combined prediction dataframes for each model type
# model_types <- c("pspline", "cyclic", "thinplate_cyclic")
# model_labels <- c("P-Spline", "Cyclic", "Thin Plate Cyclic")

# # Create a list to store combined predictions for each model type
# combined_predictions_by_model <- list()

# for (i in 1:length(model_types)) {
#   model_type <- model_types[i]
#   model_label <- model_labels[i]
  
#   combined_pred <- data.frame(Hour = hour_seq)
  
#   # Extract predictions for each behavior (excluding Uncertain and Inactive)
#   for (behavior in behaviors) {
#     if (behavior %in% c("Uncertain", "Inactive")) next
    
#     # Get predictions for this model type
#     pred <- predict(gam_models[[paste0(behavior, "_", model_type)]], 
#                    newdata = data.frame(HourMid = hour_seq), 
#                    se.fit = TRUE)
    
#     # Add to combined prediction dataframe
#     combined_pred[[behavior]] <- pred$fit
#   }
  
#   # Convert to long format and add model type
#   combined_pred_long <- combined_pred %>%
#     pivot_longer(cols = -Hour,
#                  names_to = "Behavior",
#                  values_to = "Proportion") %>%
#     mutate(Model = model_label)
  
#   combined_predictions_by_model[[model_type]] <- combined_pred_long
# }

# # Combine all model predictions
# all_model_predictions <- bind_rows(combined_predictions_by_model)

# # Create the comparison plot
# model_comparison_plot <- ggplot(all_model_predictions,
#                                aes(x = Hour, y = Proportion, color = Behavior)) +
#   geom_line(linewidth = 2, alpha = 0.8) +
#   facet_wrap(~ Model, ncol = 2, scales = "free_y") +
#   theme_bw() +
#   theme(
#     panel.grid.major = element_blank(),
#     panel.grid.minor = element_blank(),
#     panel.background = element_rect(fill = "white"),
#     axis.line = element_line(color = "black"),
#     legend.position = "right",
#     legend.title = element_text(face = "bold"),
#     plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
#     strip.text = element_text(face = "bold")
#   ) +
#   labs(title = "Daily Activity Patterns - Model Comparison",
#        x = "Hour of Day",
#        y = "Proportion",
#        color = "Behavior") +
#   scale_color_brewer(palette = "Set1") +
#   scale_x_continuous(breaks = seq(0, 24, by = 6),
#                     limits = c(0, 24))

# # Save the model comparison plot
# ggsave("Model_Comparison_All_Behaviors.png", model_comparison_plot, 
#        width = 16, height = 12, dpi = 300)

# # Create a focused comparison plot using only the best performing model
# best_models_by_behavior <- comparison_results %>%
#   select(Behavior, Best_Model) %>%
#   filter(!Behavior %in% c("Uncertain", "Inactive"))

# # Create prediction data using best models for each behavior
# best_model_predictions <- data.frame(Hour = hour_seq)

# for (i in 1:nrow(best_models_by_behavior)) {
#   behavior <- best_models_by_behavior$Behavior[i]
#   best_model_name <- best_models_by_behavior$Best_Model[i]
  
#   # Convert model name to our naming convention
#   model_suffix <- case_when(
#     best_model_name == "P-Spline" ~ "pspline",
#     best_model_name == "Cyclic" ~ "cyclic",
#     best_model_name == "Thin Plate Cyclic" ~ "thinplate_cyclic"
#   )
  
#   # Get predictions
#   pred <- predict(gam_models[[paste0(behavior, "_", model_suffix)]], 
#                  newdata = data.frame(HourMid = hour_seq), 
#                  se.fit = TRUE)
  
#   best_model_predictions[[behavior]] <- pred$fit
# }

# # Convert to long format
# best_model_predictions_long <- best_model_predictions %>%
#   pivot_longer(cols = -Hour,
#                names_to = "Behavior",
#                values_to = "Proportion")

# # Create the best models plot
# best_models_plot <- ggplot(best_model_predictions_long,
#                           aes(x = Hour, y = Proportion, color = Behavior)) +
#   geom_line(size = 1.2) +
#   theme_bw() +
#   theme(
#     panel.grid.major = element_blank(),
#     panel.grid.minor = element_blank(),
#     panel.background = element_rect(fill = "white"),
#     axis.line = element_line(color = "black"),
#     legend.position = "right",
#     legend.title = element_text(face = "bold"),
#     plot.title = element_text(face = "bold", size = 14, hjust = 0.5)
#   ) +
#   labs(title = "Daily Activity Patterns - Best Models by AIC",
#        x = "Hour of Day",
#        y = "Proportion",
#        color = "Behavior") +
#   scale_color_brewer(palette = "Set1") +
#   scale_x_continuous(breaks = seq(0, 24, by = 3),
#                     limits = c(0, 24))

# # Save the best models plot
# ggsave("Best_Models_Combined_Behaviors.png", best_models_plot, 
#        width = 12, height = 8, dpi = 300)


# ### Now add confidence intervals - and only focus on best model (thin plate cyclic) ###

# ############ Individual Thin Plate Cyclic Plots with 95% Confidence Intervals ############

# # Define the behaviors we want to plot
# target_behaviors <- c("Swimming", "Active", "Nursing", "Sleeping")

# # Generate predictions with confidence intervals for each behavior
# hour_seq_fine <- seq(0, 24, length.out = 200)  # Higher resolution for smooth curves
# prediction_data_fine <- data.frame(HourMid = hour_seq_fine)

# for (behavior in target_behaviors) {
#   # Get the original data for this behavior
#   behavior_data <- behavior_long %>% filter(Behavior == behavior)
  
#   # Get predictions with confidence intervals
#   pred_thinplate_cyclic <- predict(gam_models[[paste0(behavior, "_thinplate_cyclic")]], 
#                                   newdata = prediction_data_fine, 
#                                   se.fit = TRUE)
  
#   # Create prediction dataframe
#   pred_df <- data.frame(
#     Hour = hour_seq_fine,
#     Fit = pred_thinplate_cyclic$fit,
#     SE = pred_thinplate_cyclic$se.fit,
#     Lower = pred_thinplate_cyclic$fit - 1.96 * pred_thinplate_cyclic$se.fit,
#     Upper = pred_thinplate_cyclic$fit + 1.96 * pred_thinplate_cyclic$se.fit
#   )
  
#   # Create the plot
#   p <- ggplot() +
#     # Add confidence interval ribbon
#     geom_ribbon(data = pred_df,
#                 aes(x = Hour, ymin = Lower, ymax = Upper),
#                 alpha = 0.3, fill = "steelblue") +
#     # Add the main fitted line
#     geom_line(data = pred_df,
#               aes(x = Hour, y = Fit),
#               color = "steelblue", linewidth = 1.5) +
#     # Add the original data points
#     #geom_point(data = behavior_data,
#      #          aes(x = HourMid, y = Proportion),
#       #         color = "black", alpha = 0.6, size = 2) +
#     # Styling
#     theme_classic() +
#     theme(
#       panel.background = element_rect(fill = "white"),
#       plot.background = element_rect(fill = "white"),
#       axis.line = element_line(color = "black"),
#       axis.text = element_text(size = 12),
#       axis.title = element_text(size = 14, face = "bold"),
#       plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
#       panel.grid.major = element_blank(),
#       panel.grid.minor = element_blank()
#     ) +
#     # Labels
#     labs(title = paste("Daily", behavior, "Pattern"),
#          subtitle = "Thin Plate Cyclic Spline with 95% Confidence Interval",
#          x = "Hour of Day",
#          y = "Proportion of Time") +
#     # Scale settings
#     scale_x_continuous(breaks = seq(0, 24, by = 3),
#                       limits = c(0, 24)) +
#     scale_y_continuous(labels = scales::percent_format(accuracy = 0.1))
  
#   # Save individual plot
#   ggsave(paste0("ThinPlate_", behavior, "_with_CI.png"), p, 
#          width = 10, height = 6, dpi = 300)
  
#   # Print some model info
#   cat("\n", behavior, "Thin Plate Cyclic Model:\n")
#   cat("AIC:", round(AIC(gam_models[[paste0(behavior, "_thinplate_cyclic")]]), 3), "\n")
#   cat("Deviance Explained:", 
#       round(summary(gam_models[[paste0(behavior, "_thinplate_cyclic")]])$dev.expl * 100, 1), "%\n")
# }

# # Create a combined 2x2 panel plot
# #install.packages("gridExtra")
# library(gridExtra)
# library(grid)

# plot_list <- list()

# for (i in 1:length(target_behaviors)) {
#   behavior <- target_behaviors[i]
  
#   # Get the original data for this behavior
#   behavior_data <- behavior_long %>% filter(Behavior == behavior)
  
#   # Get predictions with confidence intervals
#   pred_thinplate_cyclic <- predict(gam_models[[paste0(behavior, "_thinplate_cyclic")]], 
#                                   newdata = prediction_data_fine, 
#                                   se.fit = TRUE)
  
#   # Create prediction dataframe
#   pred_df <- data.frame(
#     Hour = hour_seq_fine,
#     Fit = pred_thinplate_cyclic$fit,
#     SE = pred_thinplate_cyclic$se.fit,
#     Lower = pred_thinplate_cyclic$fit - 1.96 * pred_thinplate_cyclic$se.fit,
#     Upper = pred_thinplate_cyclic$fit + 1.96 * pred_thinplate_cyclic$se.fit
#   )
  
#   # Create individual plot for the grid
#   p <- ggplot() +
#     geom_ribbon(data = pred_df,
#                 aes(x = Hour, ymin = Lower, ymax = Upper),
#                 alpha = 0.3, fill = "steelblue") +
#     geom_line(data = pred_df,
#               aes(x = Hour, y = Fit),
#               color = "steelblue", linewidth = 1.2) +
#     #geom_point(data = behavior_data,
#      #          aes(x = HourMid, y = Proportion),
#       #         color = "black", alpha = 0.6, size = 1.5) +
#     theme_classic() +
#     theme(
#       panel.background = element_rect(fill = "white"),
#       plot.background = element_rect(fill = "white"),
#       axis.line = element_line(color = "black"),
#       axis.text = element_text(size = 10),
#       axis.title = element_text(size = 11, face = "bold"),
#       plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
#       panel.grid.major = element_blank(),
#       panel.grid.minor = element_blank()
#     ) +
#     labs(title = behavior,
#          x = "Hour of Day",
#          y = "Proportion") +
#     scale_x_continuous(breaks = seq(0, 24, by = 6),
#                       limits = c(0, 24)) +
#     scale_y_continuous(labels = scales::percent_format(accuracy = 0.1))
  
#   plot_list[[i]] <- p
# }

# # Combine into 2x2 grid
# combined_panel <- grid.arrange(grobs = plot_list, ncol = 2, nrow = 2,
#                               top = textGrob("Daily Behavior Patterns - Thin Plate Cyclic Splines with 95% CI", 
#                                            gp = gpar(fontsize = 16, fontface = "bold")))

# # Save the combined panel
# ggsave("Combined_ThinPlate_Behaviors_CI.png", combined_panel, 
#        width = 14, height = 10, dpi = 300)


## just need to add random effects for Pup_ID and rookery? ##
## also look at different individual pups ##
# this is all only BB right now! #
# 30 minute bins - proportions averaged. Try smaller bin? May overfit. #


