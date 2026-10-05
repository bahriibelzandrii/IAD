
#install.packages("pvclust") 
install.packages("dendextend")
library(pvclust)

data=data.frame(c(3,4,7,4,3,3,4,4,7,11,6,6,6,3,4,4,4,4))#створили масив даних
matr_D=dist(data, method = "euclidean")
tree=hclust(matr_D, method = "average") #method = "complete"
plot(tree)

rect.hclust(tree, k= 5, border=4) #виділенння боксами

hc_compl <- hclust(matr_D, method = "complete")
ph_compl <- as.phylo(hc_compl)
plot(ph_compl, type = "phylogram", cex = 0.7)
axisPhylo()

cl=kmeans(matr_D,3,10)# кластеризація за методом к-середніх
cl$cluster



# кластеризація за допомогою засобів library(pvclust)
#використ стандартний набір даних
data(Boston, package = "MASS")
set.seed(123)
#  Бутстреп дерев та розрахунок BP- и AU-ймовірностей для вузлів,
boston.pv <- pvclust(Boston, nboot = 100, method.dist = "euclidean", 
                     method.hclust = "average", quiet = TRUE)# method.dist - метод, за яким визначаємо відстань між даними, 
# method.hclust - методика кластеризації , average - це метод к-середніх
plot(boston.pv)  #графічно будуємо кластери (дендрограма)
pvrect(boston.pv, alpha=.95, border = 'green') #col=2 #border = 3 #виділенння боксами


Boston <- t(Boston) #транспонуємо таблицю
kmeans(Boston,3,10)# кластеризація за методом к-середніх, евклідова відстань

# cl=kmeans(Boston,3,10)# кластеризація за методом к-середніх
cl$cluster#вивід кластерів


#використ дані з файлу stocks.csv

getwd()
setwd("S:/student/Kit") 

x <- read.table(file = "stocks.csv", sep=',', header=TRUE)

x <- log(x) #логарифмуємо ціни
x <- x[-1] #Вилучаємо його з масиву  стовпець з датою.
x <- apply(x, 2, diff) #обчислюємо різницю між послідовними елементами


x.pv <- pvclust(x, nboot = 100, method.dist = "cor", method.hclust = "average", quiet = TRUE) # method.dist - метод, за яким визначаємо відстань між даними, 
plot(x.pv)
pvrect(x.pv, alpha=.95) #виділенння боксами



x <- t(x) #транспонуємо таблицю
cl=kmeans(x, 5, 1000) # кластеризація за методом к-середніх,розбиваємо на 5 кластерів, максимум 1000 ітерацій.
#cl=kmeans(x, 5, 100000) # розбиваємо на 5 кластерів, максимум 100000 ітерацій.
cl$cluster #вивід кластерів






#aggregate(x,by=list(cl$cluster),FUN=mean)
#x <- data.frame(x, cl$clusterr)


#https://varmara.github.io/proteomics-course/03_classification.html
#https://edu.kpfu.ru/course/view.php?id=833