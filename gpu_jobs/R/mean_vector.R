# Load necessary library 
library(gpuR)


# Create a list of random vectors (reduce size for testing)
vectors <- replicate(500, rnorm(500), simplify = FALSE)

# Define a function to compute the mean of a vector using GPU
mean_vector_gpu <- function(vec) {
  gpu_vec <- gpuVector(vec, type = "float")  # Use gpuVector
  mean_value <- mean(gpu_vec)  # Compute mean using GPU
  return(as.numeric(mean_value))
}

# Compute the means of the vectors using GPU
start_time <- Sys.time()
means_gpu <- sapply(vectors, mean_vector_gpu)
end_time <- Sys.time()

# Print the execution time
cat("GPU execution time:", end_time - start_time, "\n")
