setwd("C:/Users/teacher/Desktop/kingd")
kings<-scan("kings.txt")
kingsts <- ts(kings, start = 1, end= 42, frequency = 1) #робить часовий р€д з набору даних
kingsts
plot.ts(kingsts)
#вид≥л€Їм тренд, згладжуЇм часовий р€д
install.packages("TTR")
kingstsSMA3 <- SMA(kingsts,n=9)
plot.ts(kingstsSMA3)
#тепер працюЇм з народжуван≥сттю в 
babyboom<-scan("babyboom.txt")
str(babyboom)
babyboom
baby.ts <- ts(babyboom, frequency=12, start=c(1946,1))
plot.ts(baby.ts)
babySMA3 <- SMA(baby,n=11)
plot.ts(babySMA3)

babycomponents <- decompose(baby.ts)
plot(babycomponents)
install.packages("forecast")
library("forecast")
install.packages("tseries")
library("tseries")

L <- BoxCox.lambda(baby.ts)



short.test <- as.numeric(baby.ts) 
h <- length(short.test) 

#нейромережа
baby.nn<- nnetar(baby.ts, lambda=L, size=3) 
baby1.nn <- forecast(baby.nn, lambda=L ) 
plot(baby1.nn, include=168) 

#часовий р€д
baby.arima<- auto.arima(baby.ts, lambda=L) 
baby1.arima <- forecast(baby.arima, lambda=L) 
plot(baby1.arima, include=168) 
#h - час на €кий складаЇтьс€ прогноз

#include - к≥льк≥сть даних, €к≥ враховуютьс€ при складанн≥ прогнозу. враховуютьс€ дан≥ з к≥нц€
