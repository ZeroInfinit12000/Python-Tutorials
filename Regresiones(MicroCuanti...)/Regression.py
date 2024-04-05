import pandas_datareader as pdr
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Fechas
start=datetime.datetime(2000,1,1)
end=datetime.datetime(2020,7,1)
#Leer Symbolo y su nombre, de donde sacar los datos, start, finish
symbol={"Industry_wage_growth_m":"LCEAMN01PLM659S","IPI_ms":"POLPROINDMISMEI","IR_m":"IRSTCB01PLM156N",
        "Unempl_rate_m":"LRHUTTTTPLM156S","CPI_m":"CPALTT01PLM659N","Gross_GDP_m":"POLLORSGPTDSTSAM"}
names=list(symbol.keys())
data_id=list(symbol.values())
symbol=dict((v,k) for k,v in symbol.items())
data=pdr.DataReader(data_id,"fred",start,end)

#Change names
data=pd.DataFrame(data).rename(columns=symbol)




#Gráfico Interactivo
 
plt.plot(data)
plt.show()



#Nice plots
#data.iloc[:,0:3].plot(title="Wage",legend=True)
#----------------------------------------------------------------------------------------------

#Regresions OLS
X=data.iloc[:,1:]
Y=data.iloc[:,0]
X=sm.add_constant(X)
model_wage=sm.OLS(Y,X).fit("qr","HC3")
print(model_wage.summary(alpha=0.01))
#Otros datos de interes
print("\n\n\n\nParameter number 1: ",model_wage.params[1] )
print("Standard errors of last parameter: ", model_wage.bse[-1])
print("Predicted values mean: ", model_wage.predict().mean())
#print("Confidence intervals at 0.05", model_wage.conf_int(0.05))
#Test de la F

#1) Creas matriz que diga que parametros 1-3 son iguales a 0.
A=np.identity(len(model_wage.params))
#Primeras dos coeficientes B1 B2 son 0
A=A[1:5,:]
#print(model_wage.f_test(A))
#Lo puesdes hacer con texto tambien
print("Primeros dos parametros 0",model_wage.f_test("IPI_ms=IR_m=0"))
print("Primeros dos parametros 0",model_wage.wald_test(A,scalar=True))
#Prediciones de valores de wage al 95%
wage_predicted=model_wage.get_prediction().conf_int(alpha=0.05)
print("IPI_ms=2*IR_m+3*const",model_wage.f_test("IPI_ms=2*IR_m+3*const"))

#Prediciones concretas          
predict={"s":18.5,"IPI_ms":125,"IR_m":3,"Unemp_rate_m":7,"CPI_m":2,"Gross_GDP_m":1433}
predict=pd.DataFrame([predict.values()])
pred_result=model_wage.get_prediction(predict)
print(pred_result.conf_int(0.01))


#Mas comodo, pero mas restringido que sm simple...
#Basicamente, ~ igualdad, + meter variable, - quitar, I(X+B) es transformacion lineal
#Transformaciones logaritimcas faciles de poner
#X1*X2==X1:X2 interaccion 2 variables
#-1 quita constante
mod1=smf.ols(formula='IPI_ms~np.log(IR_m)+Gross_GDP_m + IR_m:CPI_m', data=data).fit()
print(mod1.summary())
#
#Probit:

#Y=np.random.randint(0,2,[247,1])
#mod2=sm.Probit(Y,X).fit()
#print(mod2.summary())
