setwd("C:/Users/sjker/Desktop/NFSF_data/Python Code")

library(ggplot2)
library(dplyr)
library(zoo)
library(tidyr)


## do this with each new pup merged csv.

b <- readr::read_csv("./C_merged.csv", num_threads = 8) ## Change to right ID!
#b <- readr::read_csv("0117_newseal.csv", num_threads=8) ## use for not observed seals!

### interpolate code - Do not do this though ### 
#b <- b %>%
#   mutate(date = as.Date(Behavior_UTC)) %>%
#   group_by(date) %>%
#   mutate(
#     depth = na.approx(depth, na.rm = FALSE, method = "linear"),
#     light = na.approx(light, na.rm = FALSE, method = "linear"),
#    temp = na.approx(temp, na.rm = FALSE, method = "linear")
#  ) %>%
#  ungroup()

#b <- b %>%
#  rename(depth_int = depth) # keep original depth too?

b <- b %>%
  arrange(Behavior)

b <- as.data.frame(b)

#change name to right animal!!!
write.csv(b, "C_merged.csv")
#write.csv(b, "0117_newseal_3.csv") # use for not observed seals

##################### Function to separate static and dynamic acceleration #############################

separate_accelerations <- function(file, bin_size = 3*25, num_threads=16) { # change bin size to fit window desired!!
  NFSF <- readr::read_csv(file, num_threads = num_threads) |>
    dplyr::mutate(
      # Apply static accel. filter to each axis
      Static_X = X |> zoo::rollapply(width = bin_size, FUN = mean, align = "right", fill = NA),
      Static_Y = Y |> zoo::rollapply(width = bin_size, FUN = mean, align = "right", fill = NA),
      Static_Z = Z |> zoo::rollapply(width = bin_size, FUN = mean, align = "right", fill = NA),
      # dynamic acceleration (subtract gravity)
      Dynamic_X = X - Static_X,
      Dynamic_Y = Y - Static_Y,
      Dynamic_Z = Z - Static_Z,
      # VEDBA
      VEDBA = sqrt(Dynamic_X^2 + Dynamic_Y^2 + Dynamic_Z^2)
    )
  
  return(NFSF)
}
C <- separate_accelerations("./C_merged.csv", bin_size = 3*25, num_threads = 16)

#################################### USE OPTION FOR ROLLING MEANS TO HANDLE SMALL MOVEMENTS BETTER #################################
## Uses 2 second window, center alignment, 2 second as told by adaptive window sizing
separate_accelerations <- function(file, bin_size = 2*25, num_threads=16) {
  NFSF <- readr::read_csv(file, num_threads = num_threads) |>
    dplyr::mutate(
      # Apply static accel. filter to each axis
      Static_X = X |> zoo::rollapply(width = bin_size, FUN = mean, align = "center", partial = TRUE),
      Static_Y = Y |> zoo::rollapply(width = bin_size, FUN = mean, align = "center", partial = TRUE),
      Static_Z = Z |> zoo::rollapply(width = bin_size, FUN = mean, align = "center", partial = TRUE),
      # Rest of the function remains the same
      Dynamic_X = X - Static_X,
      Dynamic_Y = Y - Static_Y,
      Dynamic_Z = Z - Static_Z,
      VEDBA = sqrt(Dynamic_X^2 + Dynamic_Y^2 + Dynamic_Z^2)
    )
  
  return(NFSF)
}
## Change to right animal ID and bin size!!! ##
C <- separate_accelerations("./C_merged.csv", bin_size = 2*25, num_threads = 16)

write.csv(C, "NFSPC_static_proc.csv")
#write.csv(d0117, "0117_newseal_3.csv.csv")



######################################## ADAPTIVE WINDOW SIZING ######################################################
#### DONT USE THIS ONE!! ###
adaptive_window <- function(data, initial_size, max_size = 5*25) {
  # Calculate rolling standard deviation
  data$magnitude <- sqrt(data$int.aX^2 + data$int.aY^2 + data$int.aZ^2)
  roll_sd <- zoo::rollapply(data$magnitude, 
                            width = initial_size, 
                            FUN = sd, 
                            align = "center", fill = NA)
  
  # Determine threshold for high variability (e.g., 75th percentile)
  sd_threshold <- quantile(roll_sd, 0.6, na.rm = TRUE)
  
  # Get the last non-NA value of roll_sd
  last_sd <- tail(na.omit(roll_sd), 1)
  
  # Adjust window size based on variability
  if (length(last_sd) > 0 && last_sd > sd_threshold) {
    return(max(initial_size / 2, 16))  # Minimum 1 second
  } else {
    return(min(initial_size * 2, max_size))  # Maximum 5 seconds
  }
}

separate_accelerations <- function(file, initial_bin_size = 1*16, num_threads = 8) {
  NFSF <- readr::read_csv(file, num_threads = num_threads)
  
  NFSF <- NFSF %>%
    dplyr::mutate(
      window_size = purrr::map_dbl(seq_along(int.aX), 
                                   ~ adaptive_window(NFSF[max(1, . - initial_bin_size + 1):., ], initial_bin_size)),
      Static_X = purrr::map2_dbl(seq_along(int.aX), window_size, 
                                 ~ mean(int.aX[max(1, .x - .y + 1):.x], na.rm = TRUE)),
      Static_Y = purrr::map2_dbl(seq_along(int.aY), window_size, 
                                 ~ mean(int.aY[max(1, .x - .y + 1):.x], na.rm = TRUE)),
      Static_Z = purrr::map2_dbl(seq_along(int.aZ), window_size, 
                                 ~ mean(int.aZ[max(1, .x - .y + 1):.x], na.rm = TRUE)),
      Dynamic_X = int.aX - Static_X,
      Dynamic_Y = int.aY - Static_Y,
      Dynamic_Z = int.aZ - Static_Z,
      VEDBA = sqrt(Dynamic_X^2 + Dynamic_Y^2 + Dynamic_Z^2)
    )
  
  return(NFSF)
}

## Change to the right animal ID!!! ##
C <- separate_accelerations("./behav_merge_0517.csv", initial_bin_size = 1*25, num_threads = 8)
d0117 <- separate_accelerations("./0117_newseal_3.csv", bin_size = 1*25, num_threads = 8)
write.csv(C, "NFSF0517_static_proc.csv")
#write.csv(d0117, "0117_newseal_3.csv.csv")

### window size of 2 seconds seems to be the best bet as trying different tresholds (0.6, 0.75, and 0.9 for SD)
# did not show any other unique window sizes besides 32 rows (2 second).

### can plot/print window sizes as you go to see what sizes were chosen 


# GO TO PYTHON - When you have python CWT code...continue here


########## Function to extract features ######################################

extract_features <- function(data, bin_size = 3*25, num_threads = 16) {
  NFSF <-  data 
  
  # Calculate the number of bins based on the bin size
  num_bins <- ceiling(nrow(NFSF) / bin_size)
  
  # List functions for calculating descriptive statistics
  fns <- list(
    sd = ~sd(.x, na.rm = TRUE),
    mean = ~mean(.x, na.rm = TRUE),
    min = ~min(.x, na.rm = TRUE),
    max = ~max(.x, na.rm = TRUE),
    med = ~median(.x, na.rm = TRUE),
    range = ~mean(range(.x, na.rm = TRUE)),
    skew = ~mean(moments::skewness(.x, na.rm = TRUE)),
    kurt = ~mean(moments::kurtosis(.x, na.rm = TRUE)),
    q25 = ~mean(quantile(.x, probs = 0.25, na.rm = TRUE)),
    q75 = ~mean(quantile(.x, probs = 0.75, na.rm = TRUE)),
    en = ~entropy::entropy(table(.x)/dplyr::n(), unit = "log2"),
    jerk = ~mean(abs(diff(.x, lag = 1, differences = 2)))
  )
  
  # Summarize data for each bin
  NFSF <- 
    NFSF |> 
    # Add grouping column based on bin
    dplyr::mutate(
      bin = rep(1:num_bins, each = bin_size, length.out = n())
    ) |> 
    # Group by bin
    dplyr::group_by(bin) |> 
    # Calculate summary statistics
    dplyr::mutate(
      dplyr::across(
        c(Static_X, Static_Y, Static_Z, Dynamic_X, Dynamic_Y, Dynamic_Z, VEDBA),
        .fns = fns,
        .names = "{col}_{.fn}"
      )
    )  
    #tidyr::drop_na() # Drop rows with NA values
  
  return(NFSF)
}

C_p <- extract_features(C, bin_size = 3*25, num_threads = 16)

############################# MOVING WINDOW APPROACH #############################################

extract_features <- function(data, bin_size = 3*25, bin_overlap = 0.25, num_threads = 16) {
  NFSF <- data
  
  # Calculate the number of bins based on the bin size and overlap
  bin_step = floor(bin_size * (1 - bin_overlap))
  num_bins <- ceiling((nrow(NFSF) - bin_size) / bin_step) + 1
  
  # List functions for calculating descriptive statistics
  fns <- list(
    sd = ~sd(.x, na.rm = TRUE),
    mean = ~mean(.x, na.rm = TRUE),
    min = ~min(.x, na.rm = TRUE),
    max = ~max(.x, na.rm = TRUE),
    med = ~median(.x, na.rm = TRUE),
    range = ~mean(range(.x, na.rm = TRUE)),
    skew = ~mean(moments::skewness(.x, na.rm = TRUE)),
    kurt = ~mean(moments::kurtosis(.x, na.rm = TRUE)),
    q25 = ~mean(quantile(.x, probs = 0.25, na.rm = TRUE)),
    q75 = ~mean(quantile(.x, probs = 0.75, na.rm = TRUE)),
    en = ~entropy::entropy(table(.x)/dplyr::n(), unit = "log2"),
    jerk = ~mean(abs(diff(.x, lag = 1, differences = 2)))
  )
  
  # Summarize data for each bin
  NFSF <- 
    NFSF |> 
    # Add grouping column based on bin
    dplyr::mutate(
      bin = rep(1:num_bins, each = bin_size, length.out = n())
    ) |> 
    # Group by bin
    dplyr::group_by(bin) |> 
    # Calculate summary statistics
    dplyr::mutate(
      dplyr::across(
        c(Static_X, Static_Y, Static_Z, Dynamic_X, Dynamic_Y, Dynamic_Z, VEDBA),
        .fns = fns,
        .names = "{col}_{.fn}"
      )
    )
  
  return(NFSF)
}

C_p <- extract_features(C_p, bin_size = 3*25, bin_overlap = 0.25, num_threads = 16)


############################ ADAPTIVE WINDOW SIZING ##############################################
extract_features_adaptive <- function(data, initial_window_size = 3*25, of = 0.5, ef = 0.2, kmax = 5) {
  NFSF <- data
  
  # List functions for calculating descriptive statistics
  fns <- list(
    sd = ~sd(.x, na.rm = TRUE),
    mean = ~mean(.x, na.rm = TRUE),
    min = ~min(.x, na.rm = TRUE),
    max = ~max(.x, na.rm = TRUE),
    med = ~median(.x, na.rm = TRUE),
    range = ~mean(range(.x, na.rm = TRUE)),
    skew = ~mean(moments::skewness(.x, na.rm = TRUE)),
    kurt = ~mean(moments::kurtosis(.x, na.rm = TRUE)),
    q25 = ~mean(quantile(.x, probs = 0.25, na.rm = TRUE)),
    q75 = ~mean(quantile(.x, probs = 0.75, na.rm = TRUE)),
    en = ~entropy::entropy(table(.x)/dplyr::n(), unit = "log2"),
    jerk = ~mean(abs(diff(.x, lag = 1, differences = 2)))
  )
  
  # Function to calculate features for a given window
  calculate_window_features <- function(window_data) {
    window_data |> 
      dplyr::summarise(
        dplyr::across(
          c(Static_X, Static_Y, Static_Z, Dynamic_X, Dynamic_Y, Dynamic_Z, VEDBA),
          .fns = fns,
          .names = "{col}_{.fn}"
        )
      )
  }
  
  # Function to classify activity (you'll need to implement this)
  classify_activity <- function(features) {
    # Implement your activity classification logic here
    # Return "transitional" or "non_transitional"
  }
  
  # Function to calculate PDF (you'll need to implement this)
  calculate_pdf <- function(features) {
    # Implement your PDF calculation logic here
    # Return the PDF value
  }
  
  results <- list()
  i <- 1
  while (i <= nrow(NFSF)) {
    k <- 0
    Ni <- initial_window_size
    
    while (k < kmax) {
      end_idx <- min(i + Ni - 1, nrow(NFSF))
      window_data <- NFSF[i:end_idx, ]
      features <- calculate_window_features(window_data)
      
      activity <- classify_activity(features)
      
      if (activity == "non_transitional") {
        pdf <- calculate_pdf(features)
        if (pdf <= pmax) {  # You'll need to define pmax
          Ni <- initial_window_size + round(ef * initial_window_size) * k
          k <- k + 1
        } else {
          break
        }
      } else {
        break
      }
    }
    
    results[[length(results) + 1]] <- list(features = features, window_size = Ni)
    i <- i + max(1, round(of * Ni))
  }
  
  # Combine results into a single data frame
  result_df <- do.call(rbind, lapply(results, function(x) {
    cbind(x$features, window_size = x$window_size)
  }))
  
  return(result_df)
}


## Change to right animal ID!! ##
C_p <- extract_features(C_p, initial_bin_size = 3*25, num_threads = 16)


# Calculate the size of each bin and Count how many bins have each size
bin_sizes <- C_p %>%
  dplyr::group_by(bin) %>%
  dplyr::summarise(bin_size = n(), .groups = 'drop')
size_counts <- bin_sizes %>%
  dplyr::count(bin_size)
#############################################################################################

C_p <- data.frame(C_p)
C_p$"Unnamed: 0" <- NULL
C_p$"...2" <- NULL
C_p$int.aX <- NULL
C_p$int.aY <- NULL
C_p$int.aZ <- NULL
C_p$Animal.ID_y <- NULL
C_p$date <- NULL
C_p$bin_size <- NULL #check this is right..and other weird cols. window_size?

########################### calculate mean features for each bin #######################

mean_values <- C_p %>%
  group_by(bin) %>%
  summarise(
    mean_light = mean(light, na.rm = TRUE),
    mean_depth = mean(depth, na.rm = TRUE),
    mean_temp = mean(temp, na.rm = TRUE),
    mean_StaticX_PF = mean(Static_X_Peak_Frequency, na.rm = TRUE),
    mean_StaticY_PF = mean(Static_Y_Peak_Frequency, na.rm = TRUE),
    mean_StaticZ_PF = mean(Static_Z_Peak_Frequency, na.rm = TRUE),
    mean_StaticX_PA = mean(Static_X_Peak_Amplitude, na.rm = TRUE),
    mean_StaticY_PA = mean(Static_Y_Peak_Amplitude, na.rm = TRUE),
    mean_StaticZ_PA = mean(Static_Z_Peak_Amplitude, na.rm = TRUE),
    mean_DynX_PF = mean(Dynamic_X_Peak_Frequency, na.rm = TRUE),
    mean_DynY_PF = mean(Dynamic_Y_Peak_Frequency, na.rm = TRUE),
    mean_DynZ_PF = mean(Dynamic_Z_Peak_Frequency, na.rm = TRUE),
    mean_DynX_PA = mean(Dynamic_X_Peak_Amplitude, na.rm = TRUE),
    mean_DynY_PA = mean(Dynamic_Y_Peak_Amplitude, na.rm = TRUE),
    mean_DynZ_PA = mean(Dynamic_Z_Peak_Amplitude, na.rm = TRUE),
    mean_VEDBA_PA = mean(VEDBA_Peak_Amplitude, na.rm = TRUE),
    mean_VEDBA_PF = mean(VEDBA_Peak_Frequency, na.rm = TRUE)
  )

C_p <- C_p %>%
  left_join(mean_values, by = "bin") %>%
  group_by(bin) %>%
  mutate(
    light = mean_light,
    depth = mean_depth,
    temp = mean_temp,
    StaticX_PF = mean_StaticX_PF,
    StaticY_PF = mean_StaticY_PF,
    StaticZ_PF = mean_StaticZ_PF,
    DynX_PF = mean_DynX_PF,
    DynY_PF = mean_DynY_PF,
    DynZ_PF = mean_DynZ_PF,
    DynX_PA = mean_DynX_PA,
    DynY_PA = mean_DynY_PA,
    DynZ_PA = mean_DynZ_PA,
    StaticX_PA = mean_StaticX_PA,
    StaticY_PA = mean_StaticY_PA,
    StaticZ_PA = mean_StaticZ_PA,
    VEDBA_PA = mean_VEDBA_PA,
    VEDBA_PF = mean_VEDBA_PF
  )


################################ SCALING ###############################################


# zero variance causes scaling to see NaNs for some columns. for 2 second bins - dont scale #


############# This code (standard scaling (z-score normalization)) below works for 3 + s bin sizes. Does not work for 0 variance cols ###########

columns_not_to_scale <- setdiff(names(C_p), c('bin','Flipper.ID', 'Accel.ID', 'Behavior','mean_depth', 'Time_UTC', 'mean_temp'))

# Scale the selected columns
scaled_columns <- as.data.frame(scale(C_p[columns_not_to_scale]))

# Combine scaled columns with the original dataframe
scaled_C_p <- C_p %>%
  select(-all_of(columns_not_to_scale)) %>%  # Remove original scaled columns
  bind_cols(scaled_columns)  # Bind the scaled columns


######################## Check for bins with multiple behavior values ###################

bins_with_multiple_behaviors <- scaled_C_p %>%
  group_by(bin) %>%
  summarize(behavior_count = n_distinct(Behavior)) %>%
  filter(behavior_count > 1) %>%
  pull(bin)

# Print the bins with multiple behaviors
print(bins_with_multiple_behaviors)

# IF THEY DO, FIX MANUALLY. Shouldnt need to do this code for the pups.

########### COMPRESS DF for one row/bin #########################################################################

scaled_C_p_comp <- scaled_C_p %>%
  group_by(bin) %>%
  summarize(across(everything(), first))

C_p_comp$light <- NULL
C_p_comp$VEDBA <- NULL
C_p_comp$Static_X <- NULL
C_p_comp$Static_Y <- NULL
C_p_comp$Static_Z <- NULL
C_p_comp$Dynamic_X <- NULL
C_p_comp$Dynamic_Y <- NULL
C_p_comp$Dynamic_Z <- NULL
C_p_comp$bin <- NULL
C_p_comp$Behavior.y <- NULL
C_p_comp$Behavior_rank <- NULL
C_p_comp$Behavior <- NULL
C_p_comp$n <- NULL
C_p_comp$pitch_angle <- NULL
C_p_comp$temp <- NULL
C_p_comp$light <- NULL
C_p_comp$depth_int <- NULL
C_p_comp$wet.dry.num <- NULL

column_names <- colnames(scaled_C_p_comp)
print(column_names)


## delete all rows that have an NA ##
scaled_C_p_comp <- na.omit(scaled_C_p_comp)

## make sure right animal ID!! ## 
write.csv(scaled_C_p_comp, "C_f.csv", row.names = FALSE)


################################## PLOTS ##############################################

# Convert Behavior_label to factor
scaled_C_p$Behavior <- factor(scaled_C_p$Behavior) # Not binned
scaled_C_p_comp$Behavior <- factor(scaled_C_p_comp$Behavior) # Binned

## this plot is how many behaviors are labeled within the bins
ggplot(scaled_C_p_comp, aes(x = Behavior)) +
  geom_bar(fill = "skyblue", color = "black", alpha = 0.5) +
  labs(x = "Behavior", y = "Frequncy within bins", title = "Bin by Behavior Type") +
  theme_minimal()

# plot of VEDBA and depth by Behavior_label
ggplot(scaled_C_p_comp, aes(x = Behavior, y = VEDBA_mean)) +
  geom_boxplot(fill = "skyblue", color = "black", alpha = 0.5) +
  labs(x = "Behavior", y = "VEDBA", title = "VEDBA by Behavior Label") +
  theme_minimal()

ggplot(scaled_C_p_comp, aes(x = Behavior, y = depth)) +
  geom_boxplot(fill = "skyblue", color = "black", alpha = 0.5) +
  labs(x = "Behavior", y = "depth (m)", title = "Depth by Behavior Label") +
  theme_minimal()

#Pitch angle plot (can be used for depth or other things too)

scaled_C_p_comp$Behavior_UTC <- as.POSIXct(scaled_C_p_comp$Behavior_UTC)

# Filter data for a specific day
start_time <- as.POSIXct("2017-09-03 00:00:00", tz = "UTC")
end_time <- as.POSIXct("2017-09-04 00:03:00", tz = "UTC")

# Subset the data for the specified day
filtered_data <- scaled_C_p_comp[scaled_C_p_comp$Behavior_UTC >= start_time & scaled_C_p_comp$Behavior_UTC < end_time, ]

# Create the plot
ggplot(filtered_data, aes(x = Behavior_UTC, y = VEDBA_mean)) +
  geom_point(alpha = 0.1, color = "skyblue") +
  geom_smooth(method = "loess", formula = y ~ x, color = "black") +
  labs(x = "Time", y = "Acceleration", title = "Acceleration by time") +
  theme_minimal()


library(hexbin)
# hexbin plot
ggplot(filtered_data, aes(x = mean_depth, y = mean_temp)) +
  geom_hex() +
  scale_fill_gradient(low = "dark blue", high = "skyblue") +
  labs(x = "Depth (m)", y = "Pitch angle (degrees)", title = "Pitch angle by Depth") +
  theme_minimal()

## faceted by behavior
ggplot(filtered_data, aes(x = Behavior_UTC, y = mean_depth)) +
  geom_line(alpha = 1, color = "blue") +
  facet_wrap(~Behavior, ncol = 3) +
  labs(x = "time", y = "depth") +
  theme_minimal()

# PLOTS FOR DEPTH W BEHAVIOR
b <- readr::read_csv("./scaled_C_p_comp.csv", num_threads = 8) ## Change to right ID!
#b = raw data before processing/binning
# can do this with processed data too
ggplot(b, aes(x = factor(Behavior), y = depth)) +
  geom_boxplot(fill = "skyblue", color = "black", alpha = 0.5) +
  labs(x = "Behavior", y = "Depth (m)", title = "Depth by Behavior Type") +
  theme_minimal()

# Example heatmap
ggplot(b, aes(x = Behavior, y = depth, fill = depth)) +
  geom_tile() +  # Heatmap
  scale_fill_gradient(low = "blue", high = "red") +  # Color scale
  labs(x = "Behavior", y = "Depth (m)", fill = "Depth") +  # Axis labels
  ggtitle("Heatmap of Depth by Behavior")  # Title

####################################### ANOVA ################################################

# Perform ANOVA
anova_result <- aov(light ~ Behavior, data = C_p)

# Perform Tukey's HSD test
tukey_results <- TukeyHSD(anova_result)

# Convert Tukey's HSD test results to a data frame
tukey_df <- broom::tidy(tukey_results)

# Filter out comparisons with p.adj > 0.05
tukey_df <- tukey_df %>%
  filter(adj.p.value > 0.05)

# Print the results
print(tukey_df)
#summary(anova_result)

######################################## PCA ###################################################

library(ggplot2)

#Only use numeric columns - 
numeric_columns <- names(C_p)[sapply(C_p, is.numeric)]
df_numeric <- C_p[, numeric_columns]
df_numeric <- na.omit(df_numeric)

# Perform PCA
pca_result <- prcomp(df_numeric, center = TRUE, scale. = FALSE)

# Extract principal component scores
pca_scores <- pca_result$x

C_p$Behavior <- as.factor(C_p$Behavior)

# Plot behaviors in 2D PCA space
ggplot(data = NULL, aes(x = pca_scores[, 1], y = pca_scores[, 2])) +
  geom_point(aes(color = C_p$Behavior, shape = C_p$Behavior)) +
  xlab("PC1") + ylab("PC2") +
  ggtitle("Behaviors in 2D PCA space")
