library(readr)
library(dplyr)
library(zoo)
library(tidyr)
library(moments)
library(entropy)

accel_dir <- "/home/dio3/williamslab/SarahKerr/valid_CSVs/"
accel_files <- list.files(path = accel_dir, pattern = "\\.csv$", full.names = TRUE)
output_dir <- "/home/dio3/williamslab/SarahKerr/Processed_Features/"

# Check which files have already been processed
processed_files <- list.files(path = output_dir, pattern = "_features\\.csv$")
processed_tags <- gsub("_features$", "", tools::file_path_sans_ext(processed_files))

# Filter out already processed files
remaining_files <- accel_files[!tools::file_path_sans_ext(basename(accel_files)) %in% processed_tags]

cat("Found", length(accel_files), "total files.\n")
cat(length(accel_files) - length(remaining_files), "files already processed.\n")
cat(length(remaining_files), "files remaining to process.\n")


#######################################################
### FIXED WINDOW APPROACH (IMPROVED)
#######################################################

separate_accelerations_fixed <- function(file, bin_size = 3*25, align = "center", delim = ";",num_threads = 16) { #could be 2 as well. Maybe use 2 for adults?
  NFSP <- readr::read_delim(file, delim=delim, num_threads = num_threads) |>
    dplyr::mutate(
      # Convert X, Y, Z to numeric first to handle potential character columns
      X = as.numeric(X),
      Y = as.numeric(Y),
      Z = as.numeric(Z)
    ) |>
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

# Process only the remaining files
for (file in remaining_files) {
  cat("Processing:", file, "\n")
  
  # Add error handling to skip problematic files
  tryCatch({
    # Step 1: Separate accelerations
    separated <- separate_accelerations_fixed(file, bin_size = 3*25, align = "center", delim = ";", num_threads = 16)
    
    # Step 2: Extract features
    extracted <- extract_features_fixed(separated, bin_size = 3*25)
    
    # Step 3: Save
    tag_id <- tools::file_path_sans_ext(basename(file))
    out_file <- file.path(output_dir, paste0(tag_id, "_features.csv"))
    write.csv(extracted, out_file, row.names = FALSE)
    
    cat("Successfully processed:", file, "\n")
  }, error = function(e) {
    cat("Error processing file:", file, "\n")
    cat("Error message:", conditionMessage(e), "\n")
    # Optionally log errors to a file
    write(paste0("Error in file: ", file, " - ", conditionMessage(e)), 
          file = file.path(output_dir, "processing_errors.log"), 
          append = TRUE)
  })
}

cat("Processing complete!\n")

## now put all 33 through CWT code.
