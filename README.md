# PyElectrochemical
Una serie de funciones escritas en python3 dedicada al análisis exploratorio de datos de experimentos electroquímicos provenientes de los aparatos Teq4 y PalmSens4

Funciones para abrir y graficar archivos de experimentos electroquímicos de voltamperometría cíclica (CV), amperometrías (Amp) y espectroscopía de impedancia electroquímica (EIS) proveniente de los aparatos Teq4 y PalmSens4.

Debido a las diferencias entre los archivos de salida que emiten los aparatos, se escribieron funciones para abrir cada uno de estos archivos con sus particularidades y almacenarlos en un dataframe de pandas.

Las rutinas se escribieron de tal modo que cada dataframe correspondiente al mismo experimento electroquímico manejen los mismos "headers", lo cual simplificó la función de graficación luego, ya que las mismas son comunes a los dos aparatos.

Estas funciones están basadas en la librería `PyEIS` de Kristian B. Knudsen (kknu@berkeley.edu || 
