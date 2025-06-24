### Preprocessing raw accel files - same exact way as merged - but do not merge  first

library(ggplot2)
library(dplyr)
library(zoo)
library(tidyr)
library(moments)
library(entropy)
library(purrr)
library(tibble)
#install.packages(c("moments", "entropy"))
# #install.packages("ssh")
# library(ssh)


## to preprocess all pups at one
#accel_dir <- "/home/dio3/williamslab/SarahKerr/AccelRaw/"
#accel_files <- list.files(path = accel_dir, pattern = "\\.csv$", full.names = TRUE)


#######################################################
### FIXED WINDOW APPROACH (IMPROVED)
#######################################################

separate_accelerations_fixed <- function(file, bin_size = 3*25, align = "center", delim = ";",num_threads = 16) { #could be 2 as well. Maybe use 2 for adults?
  NFSP <- readr::read_delim(file, delim=delim, num_threads = num_threads) |>
    dplyr::mutate(
      # Apply static accel. filter to each axis with partial=TRUE to handle edges better
      Static_X = X |> zoo::rollapply(width = bin_size, FUN = mean, align = "right", fill=NA),
      Static_Y = Y |> zoo::rollapply(width = bin_size, FUN = mean, align = "right", fill=NA),
      Static_Z = Z |> zoo::rollapply(width = bin_size, FUN = mean, align = "right", fill=NA),
      # dynamic acceleration (subtract gravity)
      Dynamic_X = X - Static_X,
      Dynamic_Y = Y - Static_Y,
      Dynamic_Z = Z - Static_Z,
      # VEDBA
      VEDBA = sqrt(Dynamic_X^2 + Dynamic_Y^2 + Dynamic_Z^2)
    )
  
  return(NFSP)
}


#######################################################
### FEATURE EXTRACTION WITH FIXED WINDOW & OVERLAP
#######################################################

extract_features_fixed <- function(data, bin_size = 3*25) {  # bin_overlap = 0. can try without overlap or larger overlap as well.
  NFSP <- data
  
  # Calculate the step size based on overlap
  #bin_step <- floor(bin_size * (1 - bin_overlap))
  
  # Calculate the number of bins based on the bin size
  num_bins <- ceiling(nrow(NFSP) / bin_size)
  
  # Pre-allocate list to store results
  # num_windows <- ceiling((nrow(NFSP) - bin_size + 1) / bin_step)
  # results <- vector("list", num_windows)
  
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
  NFSP <- 
    NFSP |> 
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
  
  return(NFSP)
}
###########################################################################
####### EXTRACT WITH AUTOCORRELATION (MOVING AVERAGES) ###### need to figure out using the GPU on server.
###########################################################################

extract_features_fixed <- function(data, bin_size = 3*25, ar_order = 3) {

  NFSP <- data
  num_bins <- ceiling(nrow(NFSP) / bin_size)

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

  vars <- c("Static_X", "Static_Y", "Static_Z", "Dynamic_X", "Dynamic_Y", "Dynamic_Z", "VEDBA")

  NFSP <- NFSP |>
    mutate(bin = rep(1:num_bins, each = bin_size, length.out = n())) |>
    group_by(bin) |>
    group_modify(~{
  df <- .x

  # Summary stats: get one-row summary for the bin
  stat_features <- df %>%
    summarise(
      across(
        .cols = all_of(vars),
        .fns = fns,
        .names = "{col}_{.fn}"
      )
    )
      # AR and MA features
      ar_ma_features <- purrr::map_dfc(vars, function(varname) {
        # Fit AR model with configurable order or AIC if ar_order=NULL
        ar_vals <- tryCatch({
          if (is.null(ar_order)) {
            ar_model <- stats::ar(df[[varname]], aic = TRUE, order.max = 5)
            order_used <- length(ar_model$ar)
          } else {
            ar_model <- stats::ar(df[[varname]], aic = FALSE, order.max = ar_order)
            order_used <- ar_order
          }

          # Prepare AR coefficient names dynamically
          coefs <- ar_model$ar
          # If model order < requested order, pad with NAs
          if(length(coefs) < order_used) {
            coefs <- c(coefs, rep(NA_real_, order_used - length(coefs)))
          }

          # Build tibble of AR coefficients
          coef_names <- paste0(varname, "_ar", 1:order_used)
          setNames(as.list(coefs), coef_names) |> tibble::as_tibble()
        }, error = function(e) {
          # If AR fit fails, return NAs for all requested lags
          coef_names <- paste0(varname, "_ar", 1:ifelse(is.null(ar_order), 5, ar_order))
          setNames(as.list(rep(NA_real_, length(coef_names))), coef_names) |> tibble::as_tibble()
        })

        # Moving averages (last value in bin)
        ma_vals <- tryCatch({
          r3 <- zoo::rollmean(df[[varname]], k = 3, fill = NA, align = "right")
          r5 <- zoo::rollmean(df[[varname]], k = 5, fill = NA, align = "right")
          tibble::tibble(
            !!paste0(varname, "_ma3") := mean(tail(r3, 1), na.rm = TRUE),
            !!paste0(varname, "_ma5") := mean(tail(r5, 1), na.rm = TRUE)
          )
        }, error = function(e) {
          tibble::tibble(
            !!paste0(varname, "_ma3") := NA_real_,
            !!paste0(varname, "_ma5") := NA_real_
          )
        })

        dplyr::bind_cols(ar_vals, ma_vals)
      })

      feature_row <- dplyr::bind_cols(stat_features[1, ], ar_ma_features)

      # Repeat features across all rows in the bin
      feature_full <- feature_row[rep(1, nrow(df)), ]
      dplyr::bind_cols(df, feature_full)
    }) |>
    ungroup()

  return(NFSP)
}

##################################################################################

# Apply the acceleration separation function
pup_processed <- separate_accelerations_fixed("/home/dio3/williamslab/SarahKerr/AccelRaw/BB_0000_S1.csv", bin_size = 3*25, align = "center", delim = ";", num_threads = 16)

# Extract features with the fixed window approach
pup_features <- extract_features_fixed(pup_processed,bin_size = 3*25, ar_order=3) #can change to larger overlap (0.25)

# Save the final dataset to put into CWT (second code is saving the one with moving averages)
write.csv(pup_features, "Pup_Features_Raw.csv", row.names = FALSE) #no behavior - raw data only to put through model and predict when behaviors are happening
write.csv(pup_features, "Pup_Features_MA.csv", row.names = FALSE)

#######################################################
### NOW MOVE TO CWT SCRIPT TO ADD IN CWT FEATURES TO THE WINDOWS
#######################################################
# after cwt features were incorporated for each bin
pup_final_features <- read.csv("final_features_cwt.csv") #change to what accel is used!
head(pup_final_features, 100)
names(pup_final_features)

# ############################# Calculate the size of each bin and Count how many bins have each size############################
#bin_sizes <- pup_final_features %>%  ##change to pup_adaptive_features for adaptive
#dplyr::group_by(bin) %>%
#dplyr::summarise(bin_size = n(), .groups = 'drop')
#size_counts <- bin_sizes %>%
#dplyr::count(bin_size)
#############################################################################################

F <- data.frame(pup_final_features) # change to A if adaptive and use pup_adaptive_feautures and go through this again
colnames(F)
F$"Temp....C." <- NULL
#F$Timestamp <- NULL Keep this to add back in after running through model.
F$X <- NULL
F$Y <- NULL
F$Z <- NULL
F$Tag.ID <- NULL
F$"Press...mBar." <- NULL
F$"ADC..raw." <- NULL
F$Time_UTC <- NULL
F$Static_X <- NULL
F$Static_Y <- NULL
F$Static_Z <- NULL
F$VEDBA <- NULL
F$Dynamic_X <- NULL
F$Dynamic_Y <- NULL
F$Dynamic_Z <- NULL
F$Metadata <- NULL
F$Batt..V...V. <- NULL
F$Accel.ID <- NULL
F$Flipper.ID <- NULL
# #any other weird cols.

# ################################ SCALING ###############################################


# # zero variance causes scaling to see NaNs for some columns. for 2 second bins - dont scale #


# ############# This code (standard scaling (z-score normalization)) below works for 3 + s bin sizes. Does not work for 0 variance cols ###########

#colnames(F) #change to A for adaptive window sizing
columns_not_to_scale <- setdiff(names(F), c('bin', 'Timestamp', 'Behavior')) #'ADC..raw.')) #adjust these accordingly!

# Scale the selected columns
scaled_columns <- as.data.frame(scale(F[columns_not_to_scale]))

# # Combine scaled columns with the original dataframe
scaled_F <- F %>%
  select(-all_of(columns_not_to_scale)) %>%  # Remove original scaled columns
   bind_cols(scaled_columns)  # Bind the scaled columns
print(scaled_F)


# ########### COMPRESS DF for one row/bin #########################################################################

#scaled_F$bin <-NULL
#write.csv(scaled_F, "FinalBB_no_comp_fixed.csv", row.names = FALSE) #change for right accel. or adaptive


# scaled_F is not compressed full df without bin column -scaled
# F_comp is compressed full df without bin column - scaled

F_comp <- scaled_F %>%
  group_by(bin) %>%
  summarize(across(everything(), first))

#colnames(F_comp)
F_comp$bin <- NULL
## delete all rows that have an NA ##
F_comp <- na.omit(F_comp)
structure(F_comp)
head(F_comp)
# ## make sure right animal ID!! ## 
write.csv(F_comp, "Final_comp_fixed.csv", row.names = FALSE) #change for adaptive (A) or right accel!


# then go to XGB code