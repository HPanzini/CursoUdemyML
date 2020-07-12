#sacar el dataset de github -andando

import pandas as pd
url = 'https://raw.githubusercontent.com/HPanzini/CursoUdemyML/master/all%20Argentina%20q2%202020.csv?token=APABFG3CRX64KWGJILDHN6C7BJOQK'
dataset = pd.read_csv(url, index_col=0, quoting=3, error_bad_lines=False)

#test raw data -andando

print(dataset.head(10))

#limpiar texto



#test de texto limpio

print(corpus)
