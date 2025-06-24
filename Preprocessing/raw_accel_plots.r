library(ggplot2)
library(dplyr)
library(zoo)
library(tidyr)
# #install.packages("ssh")
#library(ssh)
#install.packages('httpgd')
library(httpgd)
#library(lubridate)

## Need proper raw data to include the milliseconds to smooth it out...

## Plots------------------------------------------------------------------------
## Raw and filtered data plots - not binned

# Read in proper csv for the indiviual you want to look at
A <- readr::read_delim("/home/dio3/williamslab/SarahKerr/AccelRaw/A_0000_S1.csv", delim = ";", num_threads = 16)

A <- A %>%
  mutate(Timestamp = as.POSIXct(Timestamp, format = "%m/%d/%Y %H:%M:%S", tz = "UTC"))

start_time <- as.POSIXct("10/06/2024 00:00:00", format = "%m/%d/%Y %H:%M:%S", tz = "UTC")
end_time <- as.POSIXct("10/10/2024 00:00:00", format = "%m/%d/%Y %H:%M:%S", tz = "UTC")

behavior <- A %>%
  filter(Timestamp >= start_time, Timestamp <= end_time) %>%
  arrange(Timestamp) %>%
  mutate(
    time_index = row_number(),
    Timestamp_ms = Timestamp[1] + (time_index - 1) * (1 / 25)
  )

## Pulling in only certain time-frames from datasets

# low pass filter (removes high dynamic frequency)
Raw_acceleration_proccessed <- function(df, bin_size = 3 * 25) {
  df %>%
    dplyr::mutate(
      Static_X = zoo::rollapply(X, width = bin_size, FUN = mean, align = "center", fill = NA),
      Static_Y = zoo::rollapply(Y, width = bin_size, FUN = mean, align = "center", fill = NA),
      Static_Z = zoo::rollapply(Z, width = bin_size, FUN = mean, align = "center", fill = NA),
      Dynamic_X = X - Static_X, # high pass filter (removes low frequency (static))
      Dynamic_Y = Y - Static_Y,
      Dynamic_Z = Z - Static_Z,
      VEDBA = sqrt(Dynamic_X^2 + Dynamic_Y^2 + Dynamic_Z^2),
      VEDBA_smoothed = zoo::rollapply(sqrt(Dynamic_X^2 + Dynamic_Y^2 + Dynamic_Z^2),
                                width = bin_size, FUN = mean, align = "right", fill = NA)

    )
}


# Apply the function with the time range
nfs <- Raw_acceleration_proccessed(behavior, bin_size = 3 * 25)



#plotting


# raw Accelerometer data over time
nfs %>%
  ggplot(aes(x = Timestamp_ms)) +
  geom_line(aes(y = X, color = "X-Axis"), linewidth = 0.4) +
  #geom_line(aes(y = Y, color = "Y-Axis"), linewidth = 0.4) +
  #geom_line(aes(y = Z, color = "Z-Axis"), linewidth = 0.4) +
  labs(title = "Accelerometer Data over Time", x = "Time", y = "Acceleration") +
  scale_x_datetime(date_labels = "%H:%M:%S", date_breaks = "10 sec") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 80, hjust = 1, vjust = 1, size = 8)
  )


# smoothed VEDBA
nfs %>%
  ggplot(aes(x = Timestamp_ms)) +
  geom_line(aes(y = VEDBA_smoothed), color = "purple", linewidth = 0.5) +
  labs(title = "Smoothed VEDBA Over Time", x = "Time", y = "VEDBA") +
  scale_x_datetime(date_labels = "%H:%M:%S", date_breaks = "10 sec") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 80, hjust = 1, vjust = 1, size = 8)
  )


## raw stacked
nfs %>%
  dplyr::mutate(time = hms::as_hms(Timestamp_ms)) %>%
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

### stacked graph - high pass filtered (static taken out)
nfs |> 
  dplyr::mutate(time = hms::as_hms(Timestamp_ms)) |>             # convert datetime to hms
  dplyr::select(time, Dynamic_X, Dynamic_Y, Dynamic_Z, VEDBA) |> # select relevant cols
  reshape2::melt(id.vars = "time") |>                   # Melt to long format
  # Plotting separate plots for each variable
  ggplot(aes(x = time, y = value, color = variable, group = variable)) +
  geom_line()  +
  labs(title = "Dynamic Acceleration Over Time", x = "Time", y = "Acceleration") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8)
  ) +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  scale_color_manual(
    values = c("Dynamic_X" = "red", "Dynamic_Y" = "blue", "Dynamic_Z" = "green", "VEDBA"="purple"),
    labels = c("Dynamic_X" = "X-Axis", "Dynamic_Y" = "Y-Axis", "Dynamic_Z" = "Z-Axis", "VEDBA"="VEDBA")
  )


### stacked graph - low pass - smoothed (dynamic taken out)
nfs |> 
  dplyr::mutate(time = hms::as_hms(Timestamp_ms)) |>             # convert datetime to hms
  dplyr::select(time, Static_X, Static_Y, Static_Z,VEDBA_smoothed) |> # select relevant cols
  reshape2::melt(id.vars = "time") |>                   # Melt to long format
  # Plotting separate plots for each variable
  ggplot(aes(x = time, y = value, color = variable, group = variable)) +
  geom_line()  +
  labs(title = "Static Acceleration Over Time", x = "Time", y = "Acceleration") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8)
  ) +
  facet_wrap(~variable, scales = "free_y", ncol = 1) +
  scale_color_manual(
    values = c("Static_X" = "red", "Static_Y" = "blue", "Static_Z" = "green", "VEDBA"="purple"),
    labels = c("Static_X" = "X-Axis", "Static_Y" = "Y-Axis", "Static_Z" = "Z-Axis","VEDBA"="VEDBA")
  )

# static components over time (single plot)
nfs %>%
  ggplot(aes(x = Timestamp_ms)) +
  geom_line(aes(y = Static_X, color = "X-axis")) +
  geom_line(aes(y = Static_Y, color = "Y-axis")) +
  geom_line(aes(y = Static_Z, color = "Z-axis")) +
  labs(
    title = "Static Acceleration Over Time (Posture)",
    x = "Time",
    y = "Static Acceleration (g)"
  ) +
  scale_x_datetime(date_labels = "%H:%M:%S", date_breaks = "10 sec") +  # Customize time axis
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.title = element_blank()
  )


# Dynamic components over time (single plot)
nfs %>%
  ggplot(aes(x = Timestamp_ms)) +
  geom_line(aes(y = Dynamic_X, color = "X-axis")) +
  geom_line(aes(y = Dynamic_Y, color = "Y-axis")) +
  geom_line(aes(y = Dynamic_Z, color = "Z-axis")) +
  labs(
    title = "Dynamic Acceleration Over Time (Movement)",
    x = "Time",
    y = "Dynamic Acceleration (g)"
  ) +
  scale_x_datetime(date_labels = "%H:%M:%S", date_breaks = "10 sec") +
  scale_color_manual(values = c("X-axis" = "red", "Y-axis" = "green", "Z-axis" = "blue")) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.title = element_blank()
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

