
library(readr)
library(tm)
library(SnowballC)
library(skmeans)

products <- read.csv(file.choose(), header = TRUE)
clients <- read.csv(file.choose() , header = TRUE)


extract_shortname <- function(product_name) {
  # Split the name
  tokens <- strsplit(product_name, " ")[[1]]
  
  # Delete ID
  tokens <- head(tokens, length(tokens) - 1)
  
  # Delete Brands (name till the last token with digit)
  digit_indeces <- grep("[0-9]", tokens)
  
  # Product names without digits
  digit_index <- ifelse(length(digit_indeces) == 0, 1,
                        max(digit_indeces))
  paste(tokens[1:digit_index], collapse = " ")
}

# Delete product with no name
products <- products[2:nrow(products),]

products$product_shortname <- unlist(lapply(products$NombreProducto, extract_shortname))
products$product_shortname <- unlist(products$NombreProducto)

# Short Names Preprocessing
CorpusShort <- Corpus(VectorSource(products$product_shortname))
CorpusShort <- tm_map(CorpusShort, tolower)
CorpusShort <- tm_map(CorpusShort, PlainTextDocument)

# Remove Punctuation
CorpusShort <- tm_map(CorpusShort, removePunctuation)

# Remove Stopwords
CorpusShort <- tm_map(CorpusShort, removeWords, stopwords("es"))

# Stemming
CorpusShort <- tm_map(CorpusShort, stemDocument, language="es")

# Create DTM
CorpusShort <- Corpus(VectorSource(CorpusShort))
dtmShort <- DocumentTermMatrix(CorpusShort)

# Delete Sparse Terms (all the words now)
sparseShort <- removeSparseTerms(dtmShort, 0.9999)
ShortWords <- as.data.frame(as.matrix(sparseShort))

# Create valid names
colnames(ShortWords) <- make.names(colnames(ShortWords))

# Spherical k-means for product clustering (30 clusters at the moment)
set.seed(123)
mod <- skmeans(as.matrix(ShortWords), 40, method = "genetic")
products$cluster <- mod$cluster

# Example for one of the clusters
#write_csv(products[mod$cluster == 1,], 'example_cluster.csv')
write.csv(products , "E:/Business_Analytics_ISS/Kaggle/Bimbo_inventory/Products_clusters_40.csv")

library(sqldf)

unique_products <- sqldf("Select distinct NombreProducto from products")

unique_clients <- sqldf("Select distinct Cliente_ID  from clients")
