library(readr)
library(dplyr)
library(zoo)
library(tidyr)
library(moments)
library(entropy)
library(data.table)
library(vroom)

# Concatenate all 31 files together after they have all features (with CWT) and finalize the pre-processing (one file output)

# Directory and file list
input_dir <- "/home/dio3/williamslab/SarahKerr/CWT_Features/"
all_files <- list.files(input_dir, pattern = "_features_cwt\\.csv$", full.names = TRUE)

cat(sprintf("Found %d files to process\n", length(all_files)))

# Check file sizes
file_sizes <- file.info(all_files)$size
cat("File sizes (MB):\n")
for (i in seq_along(all_files)) {
  cat(sprintf("  %s: %.2f MB\n", basename(all_files[i]), file_sizes[i] / 1024^2))
}

# Read and combine function
read_and_combine_datatable <- function() {
  dfs <- lapply(seq_along(all_files), function(i) {
    cat(sprintf("[%d/%d] Reading: %s\n", i, length(all_files), basename(all_files[i])))
    tryCatch({
      fread(all_files[i], showProgress = TRUE, fill = TRUE)
    }, error = function(e) {
      cat(sprintf(" ❌ Error: %s\n", e$message))
      NULL
    })
  })
  # Remove NULLs in case of read errors
  dfs <- Filter(Negate(is.null), dfs)
  
  if (length(dfs) == 0) {
    cat("❌ No valid data frames were read\n")
    return(NULL)
  }
  
  combined_df <- rbindlist(dfs, fill = TRUE)
  return(combined_df)
}

# Combine and save
combined_df <- read_and_combine_datatable()

if (!is.null(combined_df)) {
  write_csv(as.data.frame(combined_df), "Allpups_final.csv")
  cat("✅ Saved combined file: Allpups_final.csv\n")
} else {
  cat("❌ Failed to create combined dataset\n")
}

#############################################################################################

# 1. Load your concatenated CWT feature file

#F <- fread("/home/dio3/williamslab/SarahKerr/CWT_Features/$_0000_S2_features_cwt.csv",
           showProgress = TRUE, fill= TRUE,
           nThread = 4)

#F<- read.csv("/home/dio3/williamslab/SarahKerr/CWT_Features/$_0000_S2_features_cwt.csv")
F <- fread("Allpups_final.csv", num_threads=4, showProgress = TRUE, fill =TRUE)
F <- data.frame(F)
colnames(F)
# 2. Drop unwanted columns
F <- F %>%
  select(-any_of(c(
    "X", "Y", "Z",
    "Static_X", "Static_Y", "Static_Z",
    "Dynamic_X", "Dynamic_Y", "Dynamic_Z",
    "VEDBA", "Temp....C.", "Press...mBar.",
    "Batt..V...V.", "Metadata", "Time_UTC"
  )))
#colnames(F)
# 3. Identify columns to scale (don't scale bin, timestamp, ADC, source file)
columns_not_to_scale <- c("bin", "Timestamp", "ADC..raw.", "Tag.ID")
columns_to_scale <- setdiff(names(F), columns_not_to_scale)

# 4. Apply standard scaling
scaled_columns <- as.data.frame(scale(F[ , columns_to_scale]))

# 5. Recombine scaled + unscaled columns
scaled_F <- F %>%
  select(any_of(columns_not_to_scale)) %>%
  bind_cols(scaled_columns)

# 6. Compress: keep one row per bin
F_comp <- scaled_F %>%
  group_by(bin) %>%
  summarize(across(everything(), first))

# 7. Drop bin column and any NAs
F_comp <- F_comp %>% select(-bin) %>% na.omit()
colnames(F_comp)
# 8. Save the final processed dataset
#write.csv(F_comp, "$_final.csv", row.names=FALSE)
write.csv(F_comp, "Allpups_final_comp.csv", row.names = FALSE)
print("✅ Saved: Allpups_final_comp.csv")


# now go to XGB trained model!


