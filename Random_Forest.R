# Load the required packages:
library("raster")
library("dismo")
library("tmap")
library("sf")
library("spdep")
library("rJava")
library("lwgeom")
library("ggplot2")
library("readxl")
library("zoo")
library("glmnet")
library("randomForest")
library("pROC")
library("ROCR")
# Load the UK shapefile:
UK <- read_sf("gadm41_GBR_shp/gadm41_GBR_2.shp")

# Check if there are invalid geometries and fix them:
any(st_is_valid(UK, reason = TRUE) != "Valid Geometry")
UK <- st_make_valid(UK)

# Define the extent of the shapefile:
UK_extent <- st_bbox(UK, crs = st_crs(UK))

# List of raster file types
types <- c("tmax", "prec", "vapr", "NDVI", "srad", "wind")

# Nested loop
for (type in types) {
  for (i in 1:4) {
    # Create the file path
    file_path <- paste(type, "/", type, "_Q", i, ".tif", sep = "")
    
    # Load and crop the raster
    raster_temp <- raster(file_path)
    raster_temp <- crop(raster_temp, UK_extent)
    
    # Assign raster to a new variable
    assign(paste(type, i, sep = ""), raster_temp)
  }
}
all_model_training <- list()
# Loop through all four quarters
for (quarter in 1:4) {
  
  # Load occurrence fire data in the UK for the specific quarter:
  UK_fires <- read.csv(paste("Quarter_", quarter, ".csv", sep = ""))
  coordinates(UK_fires) = ~longitude+latitude
  crs(UK_fires) <- "+proj=longlat +datum=WGS84 +no_defs"
  
  # Select the raster data for the specific quarter:
  temp <- mask(get(paste("tmax", quarter, sep = "")), UK)
  prec <- mask(get(paste("prec", quarter, sep = "")), UK)
  vapr <- mask(get(paste("vapr", quarter, sep = "")), UK)
  ndvi <- mask(get(paste("NDVI", quarter, sep = "")), UK)
  srad <- mask(get(paste("srad", quarter, sep = "")), UK)
  wind <- mask(get(paste("wind", quarter, sep = "")), UK)
  
  crs(temp) <- crs(prec)
  ndvi <- resample(ndvi, prec)
  
  envCovariates <- stack(temp, prec, vapr, ndvi, srad, wind)
  names(envCovariates) <- c("Temperature", "Precipitation", "water vapor pressure", "NDVI", "solar radiation", "Wind")
  
  # Set a seed for reproducibility
  set.seed(123456)
  
  # Convert the UK shapefile to a Spatial object for use with the spsample() function
  UK_sp <- as(UK, Class = "Spatial")
  
  # Generate a set of random background points that is twice the number of fire occurrences
  background_points <- spsample(UK_sp, n=2*length(UK_fires), "random")
  
  # Extract the environmental covariates onto the fire occurrence points and background points
  UK_fires_env <- extract(envCovariates, UK_fires)
  background_points_env <- extract(envCovariates, background_points)
  
  # Convert to data frames and add a binary indicator for fire occurrence
  UK_fires_env <-data.frame(UK_fires_env,fire=1)
  background_points_env <-data.frame(background_points_env,fire=0)
  
  # View the first few lines of the data frames
  head(UK_fires_env)
  head(background_points_env)
  
  # Use the k-fold function to split the data into 4 equal parts
  set.seed(123456)
  select <- kfold(UK_fires_env, 4)
  
  # Use 25% of the fire data for testing the model
  UK_fires_env_test <- UK_fires_env[select==1,]
  
  # Use the remaining 75% of the fire data for training the model
  UK_fires_env_train <- UK_fires_env[select!=1,]
  
  # Repeat the process for the background points
  select <- kfold(background_points_env, 4)
  background_points_env_test <- background_points_env[select==1,]
  background_points_env_train <- background_points_env[select!=1,]
  
  # Combine the fire and background data
  training_data <- rbind(UK_fires_env_train, background_points_env_train)
  testing_data <- rbind(UK_fires_env_test, background_points_env_test)
  
  # Mean imputation
  for(col in colnames(testing_data)){
    testing_data[is.na(testing_data[,col]), col] <- mean(testing_data[,col], na.rm = TRUE)
  }
  
  # Remove rows with NA values
  training_data <- na.omit(training_data)
  
  # Train the Random Forest model
  rf_model <- randomForest(as.factor(fire) ~ ., data = training_data)
  
  # Predict the class for the test set
  rf_prediction <- predict(rf_model, newdata = testing_data, type = "response")
  
  # Convert factor predictions to numeric
  rf_prediction_numeric <- as.numeric(levels(rf_prediction))[rf_prediction]
  
  # Calculate the ROC curve and the AUC
  rf_ROCR_pred <- prediction(rf_prediction_numeric, as.numeric(as.character(testing_data$fire)))
  rf_ROCR_perf <- performance(rf_ROCR_pred, "tpr", "fpr")
  
  # Calculate the AUC
  auc_ROCR <- performance(rf_ROCR_pred, measure = "auc")
  auc_ROCR <- auc_ROCR@y.values[[1]]
  
  # Predict the probability of fire occurrences
  rf_prob_fire <- predict(rf_model, newdata = as.data.frame(envCovariates), type = "prob")[,2]
  
  # Create a raster from the prediction
  rf_prob_fire_raster <- rasterFromXYZ(cbind(coordinates(envCovariates), rf_prob_fire))
  
  # Set the CRS to match the original data
  crs(rf_prob_fire_raster) <- crs(envCovariates)
  
  # Mask the probabilities:
  rf_prob_fire_raster <- mask(rf_prob_fire_raster, UK)
  
  all_model_training[[quarter]] <- rf_model
  # Create a palette of shades of red
  my_palette <- colorRampPalette(c("white", "red"))(10)
  # Plot the predicted probabilities
  if (!is.null(rf_prob_fire_raster)) {
    map1 <- tm_shape(mask(rf_prob_fire_raster, UK)) +
      tm_raster(title = "Probability", palette = my_palette, style = 'pretty', n = 10) +
      tm_shape(UK) + tm_borders(lwd = 1, col = "black") +
      tm_compass(type = "8star", size = 3, position = c("right", "top")) +
      tm_scale_bar(position = c("right", "bottom")) +  # Move scale bar
      tm_layout(main.title = paste("Predicted Wildfire in UK (Q", quarter, ")", sep = ""),
                legend.title.size = 1.2, legend.text.size = 1.2,
                main.title.size = 1.5, main.title.position = "center",
                frame = FALSE,  # Remove outer boundary of the map
                legend.position = c("left", "bottom"),
                legend.format = list(fun = function(x) as.character(x)))  # Show data values only in legend
    
  }
  
  tmap_save(map1, filename = paste("result2/RF_Q", quarter, ".png", sep = ""))
  
  # Loop through both types
  for (type_num in c(21, 41)) {
    
    # Load the future climate data for the specific quarter and type
    future_temp <- raster(paste(type_num, "tmax_Q", quarter, ".tif", sep = ""))
    future_prec <- raster(paste(type_num, "prec_Q", quarter, ".tif", sep = ""))
    
    # Crop and mask the future data
    future_temp <- crop(future_temp, UK_extent)
    future_temp <- mask(future_temp, UK)
    
    future_prec <- crop(future_prec, UK_extent)
    future_prec <- mask(future_prec, UK)
    
    # Create a new stack object with the future data
    future_envCovariates <- stack(future_temp, future_prec, vapr, ndvi, srad, wind)
    names(future_envCovariates) <- c("Temperature", "Precipitation", "water vapor pressure", "NDVI", "solar radiation", "Wind")
    
    # Predict the probability of fire occurrences in the future
    future_rf_prob_fire <- predict(rf_model, newdata = as.data.frame(future_envCovariates), type = "prob")[,2]
    
    # Create a raster from the prediction
    future_rf_prob_fire_raster <- rasterFromXYZ(cbind(coordinates(future_envCovariates), future_rf_prob_fire))
    
    # Set the CRS to match the original data
    crs(future_rf_prob_fire_raster) <- crs(future_envCovariates)
    
    # Mask the probabilities:
    future_rf_prob_fire_raster <- mask(future_rf_prob_fire_raster, UK)
    
    # Create a palette of shades of red
    my_palette <- colorRampPalette(c("white", "red"))(10)
    future_num<-type_num+19
    # Plot the predicted probabilities
    if (!is.null(future_rf_prob_fire_raster)) {
      map1 <- tm_shape(mask(future_rf_prob_fire_raster, UK)) +
        tm_raster(title = "Probability", palette = my_palette, style = 'pretty', n = 10) +
        tm_shape(UK) + tm_borders(lwd = 1, col = "black") +
        tm_compass(type = "8star", size = 3, position = c("right", "top")) +
        tm_scale_bar(position = c("right", "bottom")) +  # Move scale bar
        tm_layout(main.title = paste("20",type_num,"-20",future_num,"Predicted Wildfire in UK (Q", quarter, ")", sep = ""),
                  legend.title.size = 1.2, legend.text.size = 1.2,
                  main.title.size = 1.5, main.title.position = "center",
                  frame = FALSE,  # Remove outer boundary of the map
                  legend.position = c("left", "bottom"),
                  legend.format = list(fun = function(x) as.character(x)))  # Show data values only in legend
    }
    tmap_save(map1, filename = paste("result2/future", type_num, "_RF_Q", quarter, ".png", sep = ""))
  }
}


# Load the randomForest library
library(randomForest)

# Initialize an empty data frame to store the importance values
all_importance_df <- data.frame()

# Loop through the four quarters
for (quarter in 1:4) {
  # Extract the random forest model for the specific quarter
  rf_model <- all_model_training[[quarter]]
  
  # Compute the importance of variables
  importance_values <- importance(rf_model)
  
  # You might want to select the MeanDecreaseGini or other importance measure depending on your needs
  importance_percentage <- importance_values[, "MeanDecreaseGini"] / sum(importance_values[, "MeanDecreaseGini"]) * 100
  
  # Convert to a data frame
  importance_df <- data.frame(
    Quarter = rep(quarter, length(importance_percentage)),
    Variable = rownames(importance_values),
    Importance = importance_percentage
  )
  
  # Bind the data frames together
  all_importance_df <- rbind(all_importance_df, importance_df)
}

library(tidyverse)
library(ggplot2)
library(RColorBrewer)
# Rename the data frame to match the variable used in the ggplot code
contributions_long <- all_importance_df %>%
  rename(quarter = Quarter, contribution = Importance, variable = Variable)

# Convert quarter to a factor
contributions_long$quarter <- as.factor(contributions_long$quarter)

# Create scatter plot
ggplot(contributions_long, aes(x = contribution, y = reorder(variable, contribution), color = quarter)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Contribution of Variables by Quarter",
       subtitle = "An in-depth look into the contribution percentage by variable",
       x = "Contribution (%)",
       y = "Variable") +
  scale_color_manual(values = c("1" = "green", "2" = "red", "3" = "orange", "4" = "blue"), name = "Quarter") +
  theme_minimal() +
  theme(
    text = element_text(size = 12),
    plot.title = element_text(size = 16, face = "bold"),
    axis.title = element_text(size = 14),
    legend.title = element_text(size = 12),
    legend.position = "bottom",
    panel.grid.major = element_line(color = "grey80"),
    panel.grid.minor = element_line(color = "grey90")
  )
