# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print('Python çalışmaya başladım!!!')

a= 100

print(a)
print('iyi geceler')

import numpy
liste = [11, 12, 15]
print(numpy.std(liste, ddof=1))


import numpy as np
b = np.matrix('1 2; 3 4')
print(b)

from numpy import *  # Numpy kütüphanesini ön alan adı eki olmadan ekliyoruz
print(identity(3))     # birim köşegen matris
# birim köşegen matris
print(eye(3)) 
# köşegenin iki alt çapraza birim vektör ekleme 
print(eye(3, k=1)) 
# köşegenin iki alt çapraza birim vektör ekleme 
print(eye(3, k=-2))    
# [2,3,4] dizisinden köşegen matris
print(diag([2,3,4], k=0)) 
# aynı diziden bir üst köşegen matris
print(diag([2,3,4], k=1))
# üçlü bant matris oluşturma
print(diag(ones(4),k=-1) + diag(2*ones(5),k=0) + diag(ones(4), k=1)) 

import numpy as np  
array = [0, 1, 2, 3, 4]
np_array = np.array([0, 1, 2, 3, 4])
print(array)
print(np_array)
array + array
np_array + np_array
print(array + array)
print(np_array + np_array)
np.array([1.0, '2', '3', 4], dtype = float)
np.array([[1, 2, 3],[4, 5, 6]])
# 1 ile 5 araliginda 0.5'erlik esit adimlarla
np.arange(1, 5, 0.5)
# 0 ile 1 araliginda, 4 esit parcada
np.linspace(0, 1, 4)
#10**1 - 10**3 arasinda logaritmik olarak (** üzeri anlamindadir)
np.logspace(1, 3, 4)
# 5 elemanli her bir elemani 0 olan ve float tipinde
np.zeros(5, dtype=float)
# esas kosegeni 1 olan 3x3 boyutunda
np.eye(3)

array = np.empty(6)
array.fill(1.8)
print(array)
# matris (dizi de diyebilirsiniz) boyutunu ogrenmek icin shape kullaniyoruz
array = np.zeros((2,3,2))
print(array.shape)
# normal (gauss) dagilimda rastgele sayi (ortalamasi 0, varyansi 1)
np.random.randn(5)
# [1,10) alt sinir dahil, ust sinir dahil degil, 2 satir 3 sutunluk bir dizi
np.random.randint(1,10,(2,3))
# seed ile tohum degerini belirliyoruz.
# tohum deger diye gecen konuyu anlatmam gerekirse;
# 2 ayri bilgisayarda, ayni seed degeri ile uretilen rastgele sayilar, ayni olur.
# tohum, yani baslangiciniz ayni ise ayni sonuca (sayiya) ulasirsiniz.
# sizde eger kendi bilgisayarinizda seed(5) degerini ayarlarsaniz,
# ayni sonucu alacaksiniz.
np.random.seed(5)
np.random.randn(1)
# array: 3 satir 5 sutundan olusan 0-15 araligindaki bir dizi
array = np.arange(0,15).reshape(3,5)
print(array)
# ortalama
array.mean()
#standart sapmasi
array.std()
#varyansi
array.var()



import numpy as np
a = np.arange(15).reshape(3, 5)
print(a)
print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.itemsize)
print(a.size)
print(type(a))
b = np.array([6, 7, 8])
print(b)
print(type(b))

