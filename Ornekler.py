# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 15:10:07 2015

@author: Melek
"""
import numpy as np
a = np.arange(20).reshape(4, 5) 
print(a)
# degerleri 20'ye kadar olan 4'e 5 boyutunda biz dizi oluşturdum
b = a.shape
print(b)
# dizinin boyutunu gösterdim
c = a.ndim
print(c)
#dizinin eksenlerinin sayısını gösterdim
d = a.dtype.name
print(d)
# dizideki öğelerin türünü gösterdim
e = a.itemsize
print(e)
# dizinin ögelerinin bayt olarak boyutu gösterdim
f = a.size
print(f)
# dizinin eleman sayısını gösterdim
g = type(a)
print(g)

h = np.array([1,2,3])
print(h)
# dizi oluşturdum
i = h.dtype
print(i)
# dizinin türünü gösterdim
j = np.array([1.2, 3.4, 5.6])
print(j)
# dizi oluşturdum
k = j.dtype
print(k)
# dizinin türünü gösterdim

l = np.array([(1.5,2,3), (4,5,6)])
print(l)
# elemanları int ve float olarak oluşturduğum dizinin elemanlarını float şeklinde gösterdi

m = np.array( [ [1,2], [3,4] ], dtype=complex )
print(m)
# oluşturduğum dizinin elemanlarını complex türüne çevirdim

n = np.zeros((2,4))
print(n)
# elemanları sıfır olan dizi oluşturdum
o = np.ones((2,3,4))
print(o)
#elemanları 1 olan dizi oluşturdum
p = np.empty((2,5))
print(p)
# boş bir dizi oluşturdum

r = np.arange(12, 36, 6)
print(r)
# 12 ile 36 arasındaki sayıları 6'şar adımlarla ayırdım
s = np.arange(0, 2, 0.3)  
print(s)
# 0 ile 2 arasındaki sayıları 0.3'er adımlarla ayırdım

from numpy import pi
t = np.linspace(0, 3, 9)  
print(t)
# 0'dan 3' e kadar 9 parçaya ayırdım
u = np.linspace( 0, 2*pi, 10 )     
print(u)   
v = np.sin(u)
print(v)

y = np.arange(7)                         
print(y)
# bir boyutlu dizi oluşturdum
z = np.arange(12).reshape(4,3)          
print(z)
# iki boyutlu dizi oluşturdum
w = np.arange(24).reshape(2,3,4)         
print(w)
# üç boyutlu dizi oluşturdum

print(np.arange(10000))
# dizi çok büyükse NumPy bütün elemanlarını yazdırmaz sadece köşelerini yazdırır
print(np.arange(10000).reshape(100,100))
# diziyi 100' e 100 boyutunda yazdırdım

a = np.array( [10,20,30,40] )
b = np.arange( 4 )
print(b)
c = a-b
print(c)
d=b**2
print(d)
e=10*np.sin(a)
print(e)
f=a<25
print(f)
# oluşturduğum 2 dizi ile matematiksel işlemler yaptım

A = np.array( [[1,1],
            [0,1]] )
B = np.array( [[2,0],
            [3,4]] )
g= A*B                        
print(g)
# A ile B matrisini çarptım
x=A.dot(B)                    
print(x)
y=np.dot(A, B)                
print(y)
# matrisler arasında işlemler yaptım

h = np.ones((2,3), dtype=int)
i = np.random.random((2,3))
h *= 3
print(h)
i += h
print(i)
# elemanları 1 olan bir dizi oluşturdum
# bu dizinin 3 katını aldım
# rasgele elemanlardan oluşan diziye diğer diziyi atadım

j = np.ones(3, dtype=np.int32)
k = np.linspace(0,pi,3)
l = k.dtype.name
print(l)
m = j+k
print(m)
n = m.dtype.name
print(n)
o = np.exp(m*1j)
print(o)
p = o.dtype.name
print(p)
# int ve float olan dizilerin toplamında float şeklinde bir dizi oluşur
# dizinin türlerine baktım

r = np.random.random((3,3))
print(r)
s = r.sum()
print(s)
t = r.min()
print(t)
u = r.max()
print(u)
# rasgele degerlerden oluşan 3'e 3 boyutunda bir dizi oluşturdum
# dizinin elemanları toplamını buldum
# dizinin maximum ve minimum eleman değerini buldum

v = np.arange(12).reshape(3,4)
print(v)
y = v.sum(axis=0)   
print(y)            
# her sütun için elemanlarının toplamı buldum
z = v.min(axis=1)                           
print(z)
# her satır için minimum değeri buldum
w = v.cumsum(axis=1)                        
print(w)
# her satır için cumulative toplamu buldum

a = np.arange(3)
print(a)
b = np.exp(a)
print(b)
c = np.sqrt(a)
print(c)
d = np.array([2., -1., 3.])
print(d)
e = np.add(a, d)
print(e)
# sqrt ve exp matematik fonksiyonlarını kullandım
# d dizisini a dizisine ekledim

A = np.arange(10)**3
print(A)
# dizinin elemanlarını 3 ile çarptım
B = A[3]
print(B)
# dizinin 3.elemanını buldum
C = A[2:5]
print(C)
# dizinin 2. elemanından 5. elemanına kadar olan değerleri gösterdim

def f(x,y):
    return 10*x+y
a = np.fromfunction(f,(5,4),dtype=int)
print(a)
b = a[4,3]
print(b)
c = a[0:5, 1]                     
print(c)
d = a[ : ,1]
print(d)                        
e = a[1:3, : ]
print(e)
f = a[-1]      # son satıra eşdeğer
print(f)
# iki farklı şekilde 2.sütundaki her satırı buldum
# 2. ve 3. satırdaki her sütunları buldum

g = np.array( [[[  0,  1,  2],               
                [ 10, 12, 13]],
               [[100,101,102],
               [110,112,113]]])
h = g.shape
print(h)
i = g[1,...]    # g[1,:,:]   eşdeğer                         
print(i)
j = g[...,2]    # g[;,;,2]   eşdeğer
print(j)

for row in a:
    print(row)
# çok boyutlu dizilerde iterating işlemi birinci eksene göre yapılır.

for element in a.flat:
    print(element)
# dizinin bütün elemanları üzerinde yineleme işleme yaptım.
    
k = np.floor(10*np.random.random((3,4)))
print(k)
l = k.shape
print(l)
m = k.ravel() 
print(m)
# dizinin şeklini değiştirerek dümdüz yaptım
k.shape = (6, 2)
n = k.T
print(n)
# dizinin şeklini (6,2) olarak değiştirdim
print(k)
o = k.resize((2,6))
print(o)
# dizinin şeklini tersine çevirdim
p = k.reshape(3,-1)
print(p)

r = np.floor(10*np.random.random((2,2)))
print(r)
s = np.floor(10*np.random.random((2,2)))
print(s)
t = np.vstack((r,s))
print(t)
u = np.hstack((r,s))
print(u)
# Stack ile farklı diziler oluşturdum
# Böylece farklı eksenlerde yığılmış diziler oluşturdum

from numpy import newaxis
v = np.column_stack((r,s))   
print(v)
# 2 diziyi birlikte 2 boyutlu dizi şekline çevirdim
r = np.array([4.,2.])
print(r)
s = np.array([2.,8.])
print(s)
y = r[:,newaxis]  
print(y)
# diziyi 2 boyutlu sütun vektör şekline çevirdim
z = np.column_stack((r[:,newaxis],s[:,newaxis]))
print(z)
# 2 diziyi birlikte dizi haline getirdim
x = np.vstack((r[:,newaxis],s[:,newaxis])) 
print(x)
# sütun dizilerini birlikte sütun dizi halinde gösterdim
w = np.r_[1:5,1,0]
print(w)

a = np.floor(10*np.random.random((2,12)))
print(a)
b = np.hsplit(a,3)   
print(b)
# diziyi 3'e böldüm
c = np.hsplit(a,(3,4))   
print(c)
# diziyi 3. ve 4. sütundan sonra böldüm

d = np.arange(12)
e = d          
# yeni bir nesne oluşturdum
f = e is d         
# d ve e aynı dizi için 2 nesne
print(f)
e.shape = 3,4    
# e nin şeklini değiştirerek d'nin de şeklini değiştirmiş oldum
g = d.shape
print(g)

def f(x):
    print(id(x))
h = id(d) 
print(h)                          
i = f(d)
print(i)

j = d.view()
k = j is d
print(k)
l = j.base is d                   
print(l)
m = j.flags.owndata
print(m)
j.shape = 2,6                      
# d'nin shape' i değişmiyor
n = d.shape
print(n)
j[0,4] = 1234                      
# d'nin datası değişiyor
print(d)

o = d[ : , 1:3]    
o[:] = 10          
print(d)

p = d.copy()                         
r = p is d
print(r)
s = p.base is d                          
print(s)
p[0,0] = 9999
print(d)

t = np.arange(12)**2                      
u = np.array( [ 1,1,3,8,5 ] )              
print(t[u])                                      
# dizinin elemanlarının karesini hesaplattım
v = np.array( [ [ 3, 4], [ 9, 7 ] ] )      
print(t[v])                                       
# yine diziye atadığım değerlerin karesini hesaplattım

palette = np.array( [ [0,0,0],                # siyah
                      [255,0,0],              # kırmızı
                      [0,255,0],              # yeşil
                      [0,0,255],              # mavi
                      [255,255,255] ] )       # beyaz
image = np.array( [ [ 0, 1, 2, 0 ],           # her değer paletteki bir renge karşılık geliyor.
                     [ 0, 3, 4, 0 ]  ] )
y = palette[image]                           # (2,4,3) renk resmini gösterdim
print(y)

z = np.arange(12).reshape(3,4)
print(z)
w = np.array( [ [0,1],                        # ilk indeks
                [1,2] ] )
x = np.array( [ [2,1],                        # ikinci indeks
              [3,3] ] )
print(z[w,x])                                    
print(z[:,x])                                     
a = [w,x]
print(z[a])    

b = np.arange(5)
print(b)
# bir dizi oluşturdum
b[[1,3,4]] = 0
print(b)
# dizinin bazı elemanlarına 0 sayısını atadım

c = np.arange(5)
# bir dizi oluşturdum
c[[0,0,2]]=[1,2,3]
print(c)
# dizinin bazı elemanlarına girdiğim değerleri atadım

d = np.arange(5)
#bir dizi oluşturdum
d[[0,0,2]]+=1
print(d)
# dizinin bazı elemanlarına 1 ekledim

