#### sprawdzenie, czy cyfry się powtarzają
def sprawdz(i,j, tab):
    # test w kolumine
    for x in range(9):
        if x!=i: 
            if str(tab[x][j])==str(tab[i][j]):
                return 1
    # test w wierszu
    for y in range(9):
        if y!=j: 
            if str(tab[i][y])==str(tab[i][j]):
                return 1
    # test w pogrubionej 9 
    for a in range(3):
        for b in range(3):
            a1=a+(int((i/3))*3)
            b1=b+(int((j/3))*3)
            if a1== i and b1 == j:
                break 
            if str(tab[a+(int((i/3))*3)][b+(int((j/3))*3)])==str(tab[i][j]):
                return 1
    return 0

def solv(tabl, kopia):
    i=0
    j=0
    iter=0
    while i <9 :
        while j <9 :
            if tabl[i][j]==0:
                kopia[i][j]=1 # jeśli znaleziono pusty element wpisz 1
                while sprawdz(i,j,kopia)==1:
                    kopia[i][j]=kopia[i][j]+1 # jeśli test nie spełniony, zwiększ element o 1
                    while kopia[i][j]==10:
                        kopia[i][j]=0 # jeśli element wyszedł poza zakres, wyzeruj go i cofnij się o jeden element
                        if j==0 and i!=0:
                            i=i-1
                            j=8
                        elif j!=0:
                            j=j-1
                        while tabl[i][j]!=0:
                            if j==0 and i!=0:
                                i=i-1
                                j=8
                            elif j!=0:
                                j=j-1
                        kopia[i][j]=kopia[i][j]+1
            j=j+1
            iter=iter+1
        j=0  
        i=i+1
    return iter