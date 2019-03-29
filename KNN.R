
abstand <- function(dat1, dat2){
  
    wert <- 0
    for (i in length(dat1)) {
      t <- abs(dat1[i]-dat2[i])
      wert <- wert + t^2
    }
    return(sqrt(wert))
}

dat <- replicate(20, rnorm(10))
class <- replicate(20, rnorm(2))


mark <- c()
for (i in 1:nrow(dat)) {
  ab <- abstand(dat[i,],class[1,])
  for(j in 2:nrow(class)){
    ab <- c(abstand(dat[i,],class[j,]),ab)
  }
  mark <- c(mark, which(ab[] == min(ab)))
}

dat <- cbind(dat, mark)