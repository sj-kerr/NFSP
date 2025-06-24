setwd("C:/Users/sjker/Desktop/NFSP_data")
library("dplyr")
library(ggplot2)
library(dplyr)
library(zoo)
library(tidyr)
# #install.packages("ssh")
library(ssh)

## Need proper raw data to include the milliseconds to smooth it out...

## Plots------------------------------------------------------------------------
## Raw and filtered data plots - not binned

session <- ssh_connect("sjkerr@kndy.biology.colostate.edu")
result <- ssh_exec_internal(session, "cat /home/dio3/williamslab/SarahKerr/AccelRaw/A_0000_S1.csv") 
#raw too big to work...
 # Write to a temporary file and read
temp_file <- tempfile(fileext = ".csv")
writeBin(result$stdout, temp_file)
bigdata <- readr::read_csv(temp_file, num_threads = 16)

# Read in proper csv for the indiviual you want to look at
#NFSF0717 <- readr::read_csv("./NFSF0717.csv", num_threads = 8)

start_time <- as.POSIXct("2024-10-07 01:24:00", format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
end_time <- as.POSIXct("2024-10-07 01:24:02", format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

behavior <- bigdata %>%
  filter(Timestamp >= start_time, Timestamp <= end_time)

## Pulling in only certain time-frames from datasets

Raw_acceleration_proccessed <- function(df, bin_size = 3 * 16) {
  df %>%
    dplyr::mutate(
      Static_X = zoo::rollapply(X, width = bin_size, FUN = mean, align = "right", fill = NA),
      Static_Y = zoo::rollapply(Y, width = bin_size, FUN = mean, align = "right", fill = NA),
      Static_Z = zoo::rollapply(Z, width = bin_size, FUN = mean, align = "right", fill = NA),
      Dynamic_X = X - Static_X,
      Dynamic_Y = Y - Static_Y,
      Dynamic_Z = Z - Static_Z,
      VEDBA = sqrt(Dynamic_X^2 + Dynamic_Y^2 + Dynamic_Z^2)
    )
}


# Apply the function with the time range
nfs <- Raw_acceleration_proccessed(behavior, bin_size = 3 * 16)
# r0517 <- Raw_acceleration_proccessed(swim, bin_size = 3 * 16)
# r0118 <- Raw_acceleration_proccessed(swim, bin_size = 3 * 16)
# r0218 <- Raw_acceleration_proccessed(swim, bin_size = 3 * 16)
# r0318 <- Raw_acceleration_proccessed(swim, bin_size = 3 * 16)
# r0219 <- Raw_acceleration_proccessed(swim, bin_size = 3 * 16)
# r0319 <- Raw_acceleration_proccessed(swim, bin_size = 3 * 16)
# r0419 <- Raw_acceleration_proccessed(swim, bin_size = 3 * 16)


#plotting

library("ggplot2")

# Accelerometer data over time
nfs %>%
  ggplot(aes(x = Timestamp)) +
  geom_line(aes(y = X, color = "X-Axis"), linewidth = 0.4) +
  #geom_line(aes(y = Y, color = "Y-Axis"), linewidth = 0.4) +
  #geom_line(aes(y = Z, color = "Z-Axis"), linewidth = 0.4) +
  labs(title = "Accelerometer Data over Time", x = "Time", y = "Acceleration") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 80, hjust = 1, vjust = 1, size = 8)
  )

## stacked
nfs %>%
  dplyr::mutate(time = hms::as_hms(Timestamp)) %>%
  dplyr::select(time, X, Y, Z, VEDBA) %>%
  reshape2::melt(id.vars = "time") %>%
  ggplot(aes(x = time, y = value, color = variable)) +
  geom_line() +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  labs(title = "Raw Acceleration Over Time", x = "Time", y = "Value") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8)
  )


### stacked graph - raw

nfs |> 
  dplyr::mutate(time = hms::as_hms(Timestamp)) |>             # convert datetime to hms
  dplyr::select(time, X, Y, Z, VEDBA) |> # select relevant cols
  reshape2::melt(id.vars = "time") |>                   # Melt to long format
  # Plotting separate plots for each variable
  ggplot(aes(x = time, y = value, color = variable, group = variable)) +
  geom_line()  +
  labs(title = "Raw Acceleration Over Time", x = "Time", y = "Acceleration") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8)
  ) +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  scale_color_manual(
    values = c("int.aX" = "red", "int.aY" = "blue", "int.aZ" = "green", "VEDBA" = "purple"),
    labels = c("int.aX" = "X-Axis", "int.aY" = "Y-Axis", "int.aZ" = "Z-Axis", "VEDBA" = "VEDBA")
  )

### stacked graph - smoothed
nfs |> 
  dplyr::mutate(time = hms::as_hms(Timestamp)) |>             # convert datetime to hms
  dplyr::select(time, Dynamic_X, Dynamic_Y, Dynamic_Z) |> # select relevant cols
  reshape2::melt(id.vars = "time") |>                   # Melt to long format
  # Plotting separate plots for each variable
  ggplot(aes(x = time, y = value, color = variable, group = variable)) +
  geom_line()  +
  labs(title = "Raw Acceleration Over Time", x = "Time", y = "Acceleration") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8)
  ) +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  scale_color_manual(
    values = c("Dynamic_X" = "red", "Dynamic_Y" = "blue", "Dynamic_Z" = "green"),
    labels = c("Dynamic_X" = "X-Axis", "Dynamic_Y" = "Y-Axis", "Dynamic_Z" = "Z-Axis")
  )


### 3D plot using Plotly
#' this graph is showing VEDBA in the 3 axis plane with a gradient.
nfs |> 
  plotly::plot_ly(
    x = ~ Static_X,
    y = ~ Static_Y,
    z = ~ Static_Z,
    type = "scatter3d",
    mode = "markers",
    marker = list(size = 4, color = ~ VEDBA, colorscale = "Viridis"),
    color = ~ VEDBA,
    text = ~ paste("VEDBA: ", VEDBA)
  )  |> 
  plotly::layout(scene = list(
    aspectmode = "data",
    xaxis = list(title = "X-Axis"),
    yaxis = list(title = "Y-Axis"),
    zaxis = list(
      title = "Z-Axis",
      color = ~ VEDBA,
      cmin = min(nfs$VEDBA),
      cmax = max(nfs$VEDBA),
      colorbar = list(title = "VEDBA")
    )
  ))


### histogram showing the distribution of VEDBA values in a one minute frame 
ggplot(nfs, aes(x = VEDBA, fill = Axis)) +
  geom_histogram(binwidth = 1, position = "identity", fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Acceleration Values", x = "Acceleration", y = "Frequency") +
  theme_minimal()
# going to mainly be 0 unless the animal is actively changing acceleration
# Swimming or drifting will mainly be 0



### Heatmap
ggplot(nfs, aes(x = gmt, y = int.aX, fill = VEDBA)) +
  geom_tile() +
  labs(title = "Acceleration Heatmap", x = "Time", y = "Raw X") +
  theme_minimal()



### animated plot - good for visualizing dynamic movements
nfs |>
  dplyr::mutate(ymd_hms = lubridate::round_date(gmt, unit = "seconds")) |>
  ggplot(aes(x = ymd_hms, y = VEDBA, color = VEDBA)) +
  geom_line() +
  gganimate::transition_time(ymd_hms) +
  labs(title = "Acceleration Over Time", x = "Time", y = "Acceleration") +
  theme_minimal()


# pulling in first 10,000 rows of dataframe

# Raw_acceleration_proccessed <- function(file, bin_size=3*16){
# raw <- 
#   readr::read_csv(file, skip = 0, n_max = 10000) |> 
#   dplyr::select(-c(surface, uncorr.depth, Trip_no)) |>  # remove cols
#   dplyr::mutate(
#     Static_X = int.aX |> zoo::rollapply(width = bin_size, FUN = mean, align = "right", fill = NA),
#     Static_Y = int.aY |> zoo::rollapply(width = bin_size, FUN = mean, align = "right", fill = NA),
#     Static_Z = int.aZ |> zoo::rollapply(width = bin_size, FUN = mean, align = "right", fill = NA),
#     
#     # dynamic acceleration (subtract gravity)
#     Dynamic_X = int.aX - Static_X,
#     Dynamic_Y = int.aY - Static_Y,
#     Dynamic_Z = int.aZ - Static_Z,
#     
#     # VEDBA
#     VEDBA = sqrt(Dynamic_X^2+ Dynamic_Y^2 + Dynamic_Z^2)
#   ) 
#   return(raw)
# }
# 
# r0717<-Raw_acceleration_proccessed("./NFSF0717.csv", bin_size = 3*16)
# r0517<-Raw_acceleration_proccessed("./NFSF0517.csv", bin_size = 3*16)
# r0617<-Raw_acceleration_proccessed("./NFSF0617.csv", bin_size = 3*16)
# r0117<-Raw_acceleration_proccessed("./NFSF0117.csv", bin_size = 3*16)
# r1117<-Raw_acceleration_proccessed("./NFSF1117.csv", bin_size = 3*16)
# 
# 
# ## just using first x rows or whole dataset
# ## note, I can make a new csv file with only the date I want and graph that...
# 
# r0717 |> 
#   ggplot(aes(x = gmt)) +
#   #geom_line(aes(y = int.aY, color = "Raw-Y-Axis"), linewidth = 0.4, group =1) +
#   #geom_line(aes(y = Static_X, color = "X-Axis"), linewidth = 0.4, group =1) +
#   #geom_line(aes(y = int.aZ, color = "Raw-Z-Axis"), linewidth = 0.4, group =1) +
#   #geom_line(aes(y = Static_Y, color = "Y-Axis"), linewidth = 0.4, group =1) +
#   #geom_line(aes(y = int.aX, color = "Raw-X-Axis"), linewidth = 0.4, group =1) +
#   #geom_line(aes(y = Static_Z, color = "Z-Axis"), linewidth = 0.4, group = 1) +
#   geom_line(aes(y = VEDBA, color = "VEDBA"), linewidth = 0.4, group = 1) +
#   labs(title = "Accelerometer Data over Time", x = "Time", y = "Acceleration")  + 
#   theme(plot.title = element_text(hjust = 0.5)) +
#   theme(
#     axis.text.x = element_text(angle = 80, hjust = 1, vjust = 1, size = 8)
#   ) +
#   theme_minimal()



## Plots for wet/dry and depth

start_time <- as.POSIXct("2017-09-02 06:56:00", format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
end_time <- as.POSIXct("2017-09-02 07:00:00", format = "%Y-%m-%d %H:%M:%S", tz = "UTC")

behavior <- bigdata %>%
  filter(gmt >= start_time, gmt <= end_time)

behavior2 <- na.omit(behavior)



library(ggplot2)

# Plotting using ggplot2
ggplot(behavior2, aes(x = gmt, y = depth)) +
  #geom_line(color = "blue", linewidth = 0.5) +
  geom_line(aes(y= wet.dry.num, color = "Wet/Dry"))+
  labs(x = "Time", y = "Depth", title = "Line Graph")
ggplot(behavior2, aes(x = gmt, y = depth)) +
  geom_line(color = "blue", linewidth = 0.5) +
  #geom_line(aes(y= wet.dry.num, color = "Wet/Dry"))+
  labs(x = "Time", y = "Depth", title = "Line Graph")

##Secondary axis 
ggplot(behavior2, aes(x = gmt)) +
  geom_line(aes(y = depth*100, color = "Depth"), size = 1) +
  geom_line(aes(y = wet.dry.num, color = "Wet/Dry"), size = 1, linetype = "solid") +
  scale_y_continuous(name = "Depth", breaks = seq(0, 2, by = 1)) +
  scale_y_continuous(
    name = "Wet/Dry",
    breaks = seq(0, max(behavior2$wet.dry.num), by = 25),
    sec.axis = sec_axis(~ ./100, name = "Depth")
  ) +
  labs(x = "Time", title = "Wet/Dry vs Depth") +
  theme_minimal()

