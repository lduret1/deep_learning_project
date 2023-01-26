library(rootSolve)

nb_samples=1000000

R0=runif(nb_samples,1,5)#basic reproduction number
infectious_period=runif(nb_samples,1,10)#infectious period
X_ss=runif(nb_samples,3,10)#superspreading infectious ratio
f_ss=runif(nb_samples,0.05,0.2)#fraction of superspreading individuals at equilibrium
t=floor(runif(nb_samples,200,500))#number of tips 
s=runif(nb_samples,0.01,1)#sampling probability

#re-parametrization
removal_rate=1/infectious_period


beta_s_s= f_ss*X_ss*removal_rate*R0/(1-f_ss*(1-X_ss))
beta_n_n=removal_rate*R0-beta_s_s
beta_n_s=beta_s_s/X_ss
beta_s_n=X_ss*beta_n_n


parameters_tree=cbind(beta_n_n,beta_s_s,beta_n_s,beta_s_n,removal_rate,s,t,R0,X_ss,f_ss,infectious_period)

write.csv(parameters_tree,file="C:/Users/LORENA.LAPTOP-LGLJM15L/Documents/Cours ENS/2022-2023/deep_learning/projet/parameters_tree.csv", row.names = FALSE)
