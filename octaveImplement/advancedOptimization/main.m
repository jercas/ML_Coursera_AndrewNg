# 2017.11.13
# By JerCas Ety

options = optimset('Gradobj','on','MaxIter',100);
initialTheta = zeros(2,1);
[optTheta,functionVal,exitFlag] = fminunc(@costFunction,initialTheta,option)
