import numpy as np
from numpy import linalg as la
from numpy.linalg import svd
import math
import operator
import time
import os
from numpy.linalg import norm
from numpy import dot as dt
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
# from cur import cur_decomposition
import random
from numpy.linalg import pinv as inv
# =========================================================================================
# 											SVD
# =========================================================================================
def SVD(A,B,bias,c):
	print('inside SVD:','\n')
	U,sigmas,Vt=svd(A,full_matrices=False)
	print(len(sigmas))
	s=np.zeros((len(sigmas),len(sigmas)))
	for i in range(len(sigmas)):
		s[i][i]=sigmas[i]
	rec=U.dot(s.dot(Vt))
	rmse,mae,precision=pred(A,B,Vt,bias,c)
	print('rmse :',rmse)
	print('mae :',mae)
	print('precision :',precision)
	err=0
	square_err=0
	predictions=0
	for i in range(len(A)):
		for j in range(len(A[0])):
			# if A[i][j]!=B[i][j]:
				square_err+=(rec[i][j]-B[i][j])**2
				err+=abs(rec[i][j]-B[i][j])
				predictions+=1
	print('rmse for reconstruction: ',float(math.sqrt(square_err/predictions)))
	print('mae for reconstruction: ',float(err/predictions))

	eigens=[]
	sigma_sum=0
	for i in range(len(sigmas)):
		sigma_sum+=sigmas[i]**2
	threshold=0.9*sigma_sum
	sum=0
	last=0

	for i in range(len(sigmas)):
		sum+=sigmas[i]**2
		if(sum>threshold):
			last=i-1
			break
	k=last+1
	Vt1=np.zeros((last+1,len(Vt[0])))
	print('after 90%  energy retain :')
	rmse1,mae1,precision1=pred(A,B,Vt1,bias,c)
	print('rmse1 :',rmse1)
	print('mae1 :',mae1)
	print('precision1 :',precision1)
	return k

def pred(A,B,Vt,bias,c):
	predictions=0
	square_err=0
	err=0
	start_time=time.time()
	for i in range(len(A)):
		AV=A[i].dot(Vt.T)
		rating_for_movie=AV.dot(Vt)
		rating_for_movie=rating_for_movie

		for j in range(len(A[i])):
			
			if(B[i][j]!=0 and A[i][j]!=B[i][j]):
				predictions+=1

				err+=abs(rating_for_movie[j]-B[i][j])
				square_err+=((rating_for_movie[j]-B[i][j])**2)

				c[i][j]=rating_for_movie[j]
	
	rmse=float(math.sqrt(square_err/predictions))
	mae=float((err)/predictions)

	count=0
	k_movies_c=k_top_movies(c,k)
	
	for movie in k_movies_B:
		if movie in k_movies_c:
			count+=1

	precision=float(count)/k
	print('time taken for prediction: ',time.time()-start_time)
	return rmse,mae,precision


def k_top_movies(m,k):
	movie_rating=[]
	avg_rating=np.zeros(len(m[0]))
	k_movies_m=[]
	for j in range(len(m[0])):
		sum=0
		raters=0
		for i in range (len(m)):
			if(m[i][j]!=0):
				sum+=m[i][j]
				raters+=1
		if raters>=1:
			avg_rating[j]=float(sum)/raters
			movie_rating.append([j,avg_rating[j]])
	sorted_movies=sorted(movie_rating,key=operator.itemgetter(1),reverse=True)
	for j,ind in zip(range(k),range(len(sorted_movies))):
		k_movies_m.append(sorted_movies[j][0])
	return k_movies_m
# =========================================================================================
# 											COLLABORATIVE FILTERING
# =========================================================================================
def CF(At,Bt,c):
	print('inside CF','\n')
	# print(Bt)
	start_time=time.time()
	
	similarity=[]
	top_k=[]
	predictions=0
	square_err=0
	err=0
	print('checkpoint 1')
	for i in range(len(Bt)):
		similarity.append([])
		similarity[i]=[]
		top_k.append([])
		if(dt(Bt[i],Bt[i])==0):
			continue
		for j in range(len(Bt)):
			if(dt(Bt[j],Bt[j])==0 or i==j):
				continue
			den=math.sqrt((dt(Bt[i],Bt[i]))*(dt(Bt[j],Bt[j])))
			# print(den)
			if den==0:
				print('new movie')

				return
			sim=dt(Bt[i],Bt[j])/den
						
			similarity[i].append((j,sim))
		dec_i=sorted(similarity[i],key=operator.itemgetter(1),reverse=True)
		# print(dec_i)
		# return

		top_k[i]=[]
		
		for j,value in zip(range(k),range(len(dec_i))):
			top_k[i].append(dec_i[j][0])
	
	for m in range(len(similarity)):
		similarity[m].append((m,-2323))	
	print('chechpoint 2')
	for i in range(len(A)):
		for j in range(len(A[i])):
			if(B[i][j]!=0 and B[i][j]!=A[i][j]):
				rating=0
				n=0
				for l in range(len(top_k[j])):
					
					if(B[i][top_k[j][l]]==0):
						continue
					rating+=similarity[j][top_k[j][l]][1]*(B[i][top_k[j][l]]+bias[i])
					n+=similarity[j][top_k[j][l]][1]
				if(n==0):
					continue
				
				rate=float(rating)/n
				predictions+=1
				err+=abs(rate-(B[i][j]+bias[i]))
				square_err+=(rate-(B[i][j]+bias[i]))**2
	print('time taken for prediction: ',time.time()-start_time)
	rmse=float(math.sqrt(square_err/predictions))
	mae=float(err/predictions)
	print('rmse :', rmse)
	print('mae: ',mae)
	print('CF with baseline: ')
	sum=0
	predictions=0

	b=np.zeros(max_user+1)
	for i in range(len(B)):

		for j in range(len(B[0])):
			if(test_set[i][j]==1):
				b[i]+=B[i][j]+bias[i]
				sum+=B[i][j]+bias[i]
				predictions+=1
	bm=np.zeros(max_movie+1)
	for i in range(len(B[0])):
		for j in range(len(B)):
			if(test_set[j][i]==1):
				bm[i]+=B[j][i]+bias[j]
	for j in range(len(B[0])):
		sum=0
		for i in range(len(B)):
			if(B[i][j]!=0):
				sum+=1
		if sum==0:
			bm[j]=0
			continue
		bm[j]=bm[j]/sum

	u=float(sum)/predictions

	start_time=time.time()
	predictions=0
	square_err=0
	err=0
	for i in range(len(A)):
		for j in range(len(A[i])):
			if(test_set[i][j]==1):
				rating=0
				n=0
				bxi=u+bias[i]-u+bm[j]-u
				for l in range(len(top_k[j])):
					if(B[i][top_k[j][l]]==0):
						continue

					rating+=similarity[j][top_k[j][l]][1]*(B[i][top_k[j][l]]-(u+b[i]-u+bm[top_k[j][l]]-u))
					n+=similarity[j][top_k[j][l]][1]
				if(n==0):
					continue
				rate=bxi+float(rating)/n
				predictions+=1
				err+=abs(rate-B[i][j])
				square_err+=(rate-B[i][j])**2
	rmse=float(math.sqrt(square_err))/predictions
	mae=float(err)/predictions
	print('rmse with baseline approach: ',rmse)
	print('mae with baseline approach: ',mae)
	print('time taken for cf with baseline approach: ',time.time()-start_time)

# =========================================================================================
# 											CUR
# =========================================================================================
def decision(probability):
	return random.random()<probability
def col(A,k,row=False,eps=1):
	c=(k*np.log(k)/eps*eps)
	m,n=A.shape[0],A.shape[1]
	u,s,vh=np.linalg.svd(A,full_matrices=False)
	# print(vh)
	vh=vh[:k,:]
	# print(vh)
	probs=(1/k)*(vh**2).sum(axis=0)
	# print(probs)
	probs=[min(1,c*p)for p in probs]
	idxs=[decision(p)for p in probs]
	cols=A[:,idxs]
	# included_idx=[i for i in range(n) if idxs[i]]
	if row:
		return cols.T
	return cols
def cur(A,k,e=1,return_idx=False):
	m,n=A.shape[0],A.shape[1]
	if k>min(m,n):
		return [],[],[]
	C=col(A,k,False,eps=e)
	R=col(A.T,k,True,eps=e)
	U=inv(C)@A@inv(R)
	if return_idx:
		return C,U,R
	return C,U,R
def curf(A,k,B,bias):
	print('\n\n inside cur function: ')
	
	s,sig,v=svd(A,full_matrices=False)
	
	c,u,r=cur(A,len(sig),1,False)
	print('seg:',len(sig))
	curm=c.dot(u.dot(r))
	predictions=0
	square_err=0
	err=0
	start_time=time.time()
	
	for i in range(len(curm)):
		for j in range(len(curm[0])):
			if test_set[i][j]==1:
				predictions+=1
				# print(curm[i][j],B[i][j],'\n')
				err+=abs(curm[i][j]-B[i][j])
				square_err+=((curm[i][j]-B[i][j])**2)
	print('rmse: ',float(math.sqrt(square_err/predictions)))
	print('mae: ',float(err)/predictions)
	print('time taken for cur: ',time.time()-start_time)
	diff=0
	for i in range(len(curm)):
		for j in range(len(curm[0])):
			diff+=(abs(curm[i][j]-A[i][j])**2)
	print('reconstruction error: ',(diff))
	c,u,r=cur(A,k,1,False)
	curm=c.dot(u.dot(r))
	# print(curm)
	# print(A)
	diff=0
	for i in range(len(curm)):
		for j in range(len(curm[0])):
			diff+=(abs(curm[i][j]-A[i][j])**2)
	print('reconstruction error: ',(diff))
	predictions=0
	square_err=0
	err=0
	start_time=time.time()
	for i in range(len(curm)):
		for j in range(len(curm[0])):
			if test_set[i][j]==1:
				predictions+=1
				err+=abs(curm[i][j]-B[i][j])
				square_err+=(curm[i][j]-B[i][j])**2
	print('rmse with 90%  energy: ',float(math.sqrt(square_err/predictions)))
	print('mae with 90%  energy: ',float(err)/predictions)
	print('time taken for cur with 90%  energy: ',time.time()-start_time)

# =========================================================================================
# 											LATENT FACTOR
# =========================================================================================

def matrix_factorization(R, P, Q, K, steps=10, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k]+alpha*(2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j]+alpha*(2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j]-np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2)*( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T
def lf(A,B,bias):
	# print(A)
	p=np.random.rand(len(A),3)
	q=np.random.rand(len(A[0]),3)
	p1,q1=matrix_factorization(A,p,q,3)
	# print(p1.dot(q1.T))
	# print(A)
	print('check1')
	start_time=time.time()
	pq=p1.dot(q1.T)
	for i in range(len(pq)):
		for j in range(len(pq[0])):
			if(A[i][j]==0):
				pq[i][j]=0
	square_err=0
	err=0
	predictions=0
	# print(pq)
	# print(B)
	for i in range(len(pq)):
		for j in range(len(pq[0])):
			# if(test_set[i][j]==1):
				predictions+=1
				square_err+=(pq[i][j]-B[i][j])**2
				err+=abs(pq[i][j]-B[i][j])
	print('rmse for lf: ',float(math.sqrt(square_err/predictions)))
	print('mae for lf: ',float(err/predictions))
	print('time taken for lf: ',time.time()-start_time)


# =========================================================================================
# 											MAIN
# =========================================================================================
max_user=0
max_movie=0
count=0
f=open("C:\\Users\\home\\Desktop\\ir3\\rate.txt",'r')
for line in f:
	count+=1
	val=line.split("::")
	a=int(val[0])-1
	b=int(val[1])-1
	if(a>max_user):
		max_user=a
	if(b>max_movie):
		max_movie=b
test=int(0.90*count)
A=np.zeros((max_user+1,max_movie+1))
B=np.zeros((max_user+1,max_movie+1))
test_set=np.zeros((max_user+1,max_movie+1))
c=0
f=open("C:\\Users\\home\\Desktop\\ir3\\rate.txt",'r')
for line in f:
	
	val=line.split("::")
	a=int(val[0])
	b=int(val[1])
	B[a-1][b-1]=int(val[2])
	if(c<test):
		A[a-1][b-1]=int(val[2])
		c+=1
	else:
		test_set[a-1][b-1]=1

# print(test_set)
c=A.copy()
d=B.copy()
bias=np.zeros(max_user+1)
for i in range(max_user+1):
	sum=0
	movies_rated=0
	for j in range(max_movie+1):
		if B[i][j]!=0:
			sum+=B[i][j]
			movies_rated+=1
	if(movies_rated>=1):
		bias[i]=float(sum/movies_rated)
	for j in range(max_movie+1):
		if B[i][j]!=0:
			B[i][j]=B[i][j]-bias[i]

for i in range(max_user+1):
	sum=0
	movies_rated=0
	for j in range(max_movie+1):
		if A[i][j]!=0:
			sum+=A[i][j]
			movies_rated+=1
	if(movies_rated>=1):
		bias[i]=float(sum/movies_rated)
	for j in range(max_movie+1):
		if A[i][j]!=0:
			A[i][j]=A[i][j]-bias[i]

k=3
k_movies_B=k_top_movies(B,k)
k=SVD(A,B,bias,c)
curf(A,k,B,bias)
