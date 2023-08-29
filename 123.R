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

quarter<-4
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


# Train the Logistic Regression model
lr_model <- glm(fire ~ ., data = training_data, family = binomial)

# Evaluate the model using the test data
lr_prediction <- predict(lr_model, testing_data, type = "response")
lr_ROCR_pred <- prediction(lr_prediction, testing_data$fire)
lr_ROCR_perf <- performance(lr_ROCR_pred, "tpr", "fpr")

# Calculate the AUC
auc_ROCR <- performance(lr_ROCR_pred, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]

# Evaluate the model using the test data
cross_validation <- evaluate(p=testing_data[testing_data$fire==1,], a=testing_data[testing_data$fire==0,], model = lr_model)

# Plot the ROC curve
plot(cross_validation, 'ROC', cex=1.2)

# Predict the probability of fire occurrences
lr_prob_fire <- predict(lr_model, newdata = as.data.frame(envCovariates), type = "response")

# Create a raster from the prediction
lr_prob_fire_raster <- rasterFromXYZ(cbind(coordinates(envCovariates), lr_prob_fire))

# Set the CRS to match the original data
crs(lr_prob_fire_raster) <- crs(envCovariates)

# Mask the probabilities:
lr_prob_fire_raster <- mask(lr_prob_fire_raster, UK)

# Plot the predicted probabilities
if (!is.null(lr_prob_fire_raster)) {
  # Create a palette of shades of red
  my_palette <- colorRampPalette(c("white", "red"))(10)
  
  # Create the map
  map1 <- tm_shape(mask(prob_fire, UK)) +
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
tmap_save(map1, filename = paste("result1/lr_Q", quarter, ".png", sep = ""))