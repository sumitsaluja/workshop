# mean_vectors.R
library(foreach)
library(doParallel)

# Create a list of 1000 random vectors
vectors <- replicate(1000, rnorm(1000), simplify = FALSE)

# Define a function to compute the mean of a vector
mean_vector <- function(vec) {
  mean(vec)
}

# Compute the means of the vectors using foreach with 4 cores
cl <- makeCluster(4)
registerDoParallel(cl)
start_time <- Sys.time()
means <- foreach(vec = vectors) %dopar% mean_vector(vec)
end_time <- Sys.time()
stopCluster(cl)

# Compute the means of the vectors using a for loop
start_time_serial <- Sys.time()
means_serial <- numeric(length(vectors))
for (i in seq_along(vectors)) {
  means_serial[i] <- mean_vector(vectors[[i]])
}
end_time_serial <- Sys.time()

# Print the execution times
cat("Parallel execution time:", end_time - start_time, "\n")
cat("Serial execution time:", end_time_serial - start_time_serial, "\n")
