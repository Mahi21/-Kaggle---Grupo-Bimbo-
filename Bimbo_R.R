#### Grupo Bimbo ####

library(data.table)
library(sqldf)
library(sampling)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(neuralnet)
library(nnet)
library(grnn)
library(gbm)
library(h2o)

setwd("E:/Business_Analytics_ISS/Kaggle/Bimbo_inventory")
Bimbo <- fread("train.csv", stringsAsFactors = F)

train <- Bimbo

Bimbo_test <- fread("test.csv",stringsAsFactors = F)
Bimbo_test$Demanda_uni_equil <- NA


attach(Bimbo)
# 74,180,464 transactions


#Setting Key for Tables to merge

setkey(Agencia_ID)
setkey(Cliente_ID)
setkey(Producto_ID)

#### Data Description 

head(Bimbo , n = 25)
str(Bimbo)
summary(Bimbo)

#Missing Value Check 

missingvalue_check <- sapply(Bimbo, function(x) sum(is.na(x)))
missingvalue_check

#Data Cleaning 

Bimbo[] <- lapply(Bimbo[,1:6], factor)


### Random sample for pre-liminary model

sample <- strata(data = Bimbo , stratanames = NULL , size = 100000 , method = "srswor")
sample <- sampling::mstage(data = Bimbo , size = 100000 , method = "srswor")
                           
sample_bimbo <- getdata(Bimbo , sample)
sample_bimbo$ID_unit <- NULL
sample_bimbo$Prob <- NULL
sample_bimbo <- as.data.frame(sample_bimbo)
sample_bimbo$ID_unit <- NULL
sample_bimbo$Prob_.1._stage <- NULL

sample_bimbo[] <- lapply(sample_bimbo, factor)

sample_bimbo$Venta_hoy <- as.numeric(sample_bimbo$Venta_hoy)
sample_bimbo$Dev_proxima <- as.numeric(sample_bimbo$Dev_proxima)
sample_bimbo$Demanda_uni_equil <- as.numeric(sample_bimbo$Demanda_uni_equil)
sample_bimbo$Ruta_SAK <- as.integer(sample_bimbo$Ruta_SAK)
sample_bimbo$Cliente_ID <- as.integer(sample_bimbo$Cliente_ID)
sample_bimbo$Producto_ID <- as.integer(sample_bimbo$Producto_ID)


str(sample_bimbo)
attach(sample_bimbo)


#Decision Tree

#tree <- rpart(formula = (Demanda_uni_equil ~ Semana Agencia_ID Canal_ID  Ruta_SAK  Cliente_ID  Producto_ID) ,
#              data = sample_bimbo , method = "anova")
summary(tree)
rattle()
fancyRpartPlot(tree)
prp(tree)

net <- nnet(formula = (Demanda_uni_equil ~ Semana+Agencia_ID+Canal_ID+Ruta_SAK+Cliente_ID+Producto_ID) , 
            data = sample_bimbo)

grnet <- grnn::learn(set = sample_bimbo)
grnet_test <- guess(grnet , Bimbo_test)

gbm_sample <- gbm(formula = (Demanda_uni_equil ~ Semana+Agencia_ID+Canal_ID+Ruta_SAK+Cliente_ID+Producto_ID)
                  , data = sample_bimbo , n.trees = 500 , distribution = "poisson",
                  n.cores = 4)
gbm_sample$trees
summary(gbm_sample)

#Predict 

Predns <- predict.gbm(gbm_sample , newdata = Bimbo_test,n.trees = 300)
write.csv(Predns , "E:/Business_Analytics_ISS/Kaggle/Bimbo_inventory/Predictions2.csv")


##### Deep learning with H2O #######

h2oServer <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, 
                              max_mem_size = '8g')
h20_x <- c('Semana','Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID')
h2o_y <- Demanda_uni_equil
h2o_frame <- h2o::as.h2o(sample_bimbo , destination_frame = "h2o_frame")

h20_deep <- h2o::h2o.deeplearning(x= h20_x , 
                                  training_frame = h2o_frame,
                                  overwrite_with_best_model = TRUE,
                                  autoencoder = TRUE)

summary(h20_deep)
Bimbo_test$Agencia_ID <- as.factor(Bimbo_test$Agencia_ID)
Bimbo_test$Semana <- as.factor(Bimbo_test$Semana)
Bimbo_test$Canal_ID <- as.factor(Bimbo_test$Canal_ID)

h2o_test_frame <- h2o::as.h2o(Bimbo_test , destination_frame = "h2o_test_frame")
h2o_predns <- h2o::h2o.predict(h20_deep , newdata = h2o_test_frame)


###### New Analysis

#Data Cleaning Train

train <- train[,Semana := as.factor(Semana)]
train <- train[,Agencia_ID := as.factor(Agencia_ID)]
train <- train[,Canal_ID := as.factor(Canal_ID)]
train <- train[,Ruta_SAK := as.factor(Ruta_SAK)]
train <- train[,Cliente_ID := as.factor(Cliente_ID)]
train <- train[,Producto_ID := as.factor(Producto_ID)]

train_new <- train

#Data Cleaning Test

Bimbo_test <- Bimbo_test[,Semana := as.factor(Semana)]
Bimbo_test <- Bimbo_test[,Agencia_ID := as.factor(Agencia_ID)]
Bimbo_test <- Bimbo_test[,Canal_ID := as.factor(Canal_ID)]
Bimbo_test <- Bimbo_test[,Ruta_SAK := as.factor(Ruta_SAK)]
Bimbo_test <- Bimbo_test[,Cliente_ID := as.factor(Cliente_ID)]
Bimbo_test <- Bimbo_test[,Producto_ID := as.factor(Producto_ID)]

Bimbo_test$id <- NULL


#Merging Client , Product , Town Datasets with Train

Clients <- fread("Client_table.csv" , stringsAsFactors = T)
Products <- fread("ProductID_clusters.csv")
States <- fread("States.csv", stringsAsFactors = T)

States$V9 <- NULL
States$V10 <- NULL
States$V11 <- NULL
States$V12 <- NULL

States <- States[,Agencia_ID := as.factor(Agencia_ID)]
States <- States[, `Master Agency` := as.factor(`Master Agency`)]

train <- merge(train , Clients , by = "Cliente_ID")
train <- merge(train , Products , by = "Producto_ID")
train_new2 <- train
train <- train[,cluster := as.factor(cluster)]

train <- merge(train , States , by = "Agencia_ID")

write.csv(train , "E:/Business_Analytics_ISS/Kaggle/Bimbo_inventory/Cleaned_Training_Set/cleaned_train.csv")

#Fitting Model

##### Deep learning with H2O #######

h2oServer <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, 
                      max_mem_size = '10g')

h2o_y <- Demanda_uni_equil
h2o_frame <- h2o::as.h2o(train , destination_frame = "h2o_frame")
h2o_x <- colnames(h2o_frame)
h2o_x[!h2o_x %in% "Demanda_uni_equil"]

h20_deep <- h2o::h2o.deeplearning(x= h20_x ,
                                  training_frame = h2o_frame,
                                  overwrite_with_best_model = TRUE,
                                  autoencoder = TRUE)

summary(h20_deep)


#### Extreme Gradient Boosting

library(xgboost)

#all_data$xgb <- all_data[,-1]

data_xgboost <- as.matrix(train)
#data_xgb <- xgb.DMatrix(data_xgboost)

#train_xgb <- read.table(file.choose() , header = TRUE , sep = ",", 
#                        colClasses = c("numeric", rep("factor", 9)))

library(Matrix)
library(dummies)
library(caret)

#train_xgb <- caret::dummyVars(Demanda_uni_equil ~ ., data = train)
#data_xgb <- xgb.DMatrix(train_xgb)



#output_vector <- c(as.integer(all_data$Repeat)) #Consider 1 as repeating cust and 2 as non-repeating customers
#output_vector <- replace(output_vector , list = c(16816:433922), values = 0)


train_xgb <- train

colnames(train_xgb) <- c("AgenciaID",           "ProductoID"     ,     "ClienteID"         ,  "Semana"    ,
                         "CanalID" ,            "RutaSAK" ,            "Ventaunihoy",        "Ventahoy"    ,      
                         "Devuniproxima"  ,    "Devproxima"       ,   "Demandauniequil" ,   "NombreCliente"  ,     
                         "Type"  ,               "cluster"    ,          "State"     ,           "Clean Town"   ,       
                         "Master Agency" ,       "Agency Type"   ,       "Densitycategory" ,    "Competitioncategory",
                         "AgencyCategory")

train_xgb$NombreCliente <- NULL
sparse_matrix <- sparse.model.matrix(Demandauniequil ~ ., data = train_xgb)

matrix <- model.matrix(Demandauniequil ~ ., data = train_xgb)


sparse_matrix <- xgb.DMatrix(data =sparse_matrix , label = output_vector)

test_matrix <- sparse.model.matrix(Repeat ~ .,-1,data = test_cust)
test_matrix <- xgb.DMatrix(data = test_matrix , label = output_vector)

fit_xgboost <- xgboost(data = sparse_matrix , label = NULL,
                       params = list(eta = 0.05, nthread = 10,
                                     max.depth = 25,
                                     objective = "binary:logistic"),
                       nrounds = 100)


pred_xgboost <- predict(fit_xgboost , test_matrix)


## xgboost
model_xgb_cv <- xgb.cv(data=as.matrix(train),
                       label=as.matrix(train$Demanda_uni_equil),
                       nfold=10, 
                       objective="reg:linear", 
                       nrounds=200, 
                       eta=0.05, 
                       max_depth=6, 
                       subsample=0.75, 
                       colsample_bytree=0.8, 
                       min_child_weight=1, 
                       eval_metric="rmse")
model_xgb <- xgboost(data=as.matrix(X_train), label=as.matrix(X_target), objective="binary:logistic", nrounds=200, eta=0.05, max_depth=6, subsample=0.75, colsample_bytree=0.8, min_child_weight=1, eval_metric="auc")


### Gradient Boosting

fit_gbm <- gbm(formula = (Demanda_uni_equil ~ .)
                  , data = train , n.trees = 500 , distribution = "gaussian",
                  n.cores = 4)
gbm_sample$trees
summary(gbm_sample)


