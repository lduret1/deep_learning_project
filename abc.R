library(abc)

###importation of the summary statistics

chemin_ss="C:/Users/LORENA.LAPTOP-LGLJM15L/Documents/Cours ENS/2022-2023/deep_learning/projet/training_set"
setwd(chemin_ss)
docs_ss=list.files(chemin_ss)

data_ss=read.csv(docs_ss[1], row.names="X")

for (i in 2:length(docs_ss)){
  print(i)
  small_data_ss=read.csv(docs_ss[i], row.names="X")
  data_ss=rbind(data_ss,small_data_ss)
}

#importation of the rescaling factors

rf=c()

chemin_rf="C:/Users/LORENA.LAPTOP-LGLJM15L/Documents/Cours ENS/2022-2023/deep_learning/projet/rescaling_factors"
setwd(chemin_rf)
docs_rf=list.files(chemin_rf)

for (i in 1:length(docs_rf)){
  print(i)
  small_rf=read.csv(docs_rf[i], row.names = "X")$X0
  rf=append(rf,small_rf)
}

#importation of the parameters
chemin_param="C:/Users/LORENA.LAPTOP-LGLJM15L/Documents/Cours ENS/2022-2023/deep_learning/projet/parameters_tree.csv"
big_data_param=read.csv(chemin_param)
data_param=big_data_param[1:length(rf),seq(8,11)]

data_ss=cbind(data_ss,big_data_param[1:length(rf),c(6,7)])

#cleaning of the data
data_ss=data_ss[rf!=0,]
data_param=data_param[rf!=0,]
rf=rf[rf!=0]

data_ss=data_ss[,-c(3,20)]

#rescaling of the time dependent parameters
data_param[,4]=data_param[,4]/rf


#importation of the target
chemin_real="C:/Users/LORENA.LAPTOP-LGLJM15L/Documents/Cours ENS/2022-2023/deep_learning/projet/real_data"
setwd(chemin_real)
real_ss=read.csv("Zurich_HIV_SS.csv", row.names="X")[,-c(3,20)]
real_ss=c(real_ss,0.25,200)

#computation of abc

abc=abc(target = real_ss, param=data_param, sumstat=data_ss,method="rejection",tol=0.001)
simul_select=abc$region

#cross validation to know the goodness of the posterior distribution
set.seed(1)
cv=cv4abc(data_param,data_ss,abc,nval=100,method="rejection",tol=0.001)
summary(cv)
plot(cv)
