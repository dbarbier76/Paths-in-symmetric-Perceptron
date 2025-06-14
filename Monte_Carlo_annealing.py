import sys
import math 
import numba
import numpy as np 

from scipy import integrate, optimize, special
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

from matplotlib import pyplot as plt


#sys.path.append('./Perceptron-vs-Regularization')
#sys.path.append('./')
from Monte_Carlo_modules import d_interaction_vector_func,perceptron_cost_adapt_bias,perceptron_cost_adapt_bias2

def diag_mat(matrix,N_eigvec):
    matrix_sparse=csr_matrix(matrix)
    eigenvalue, eigenvector = eigs(matrix_sparse, k=N_eigvec, which='LM')
    return np.real(eigenvalue), np.real(eigenvector)

def Generating_clique_matrix(κ,m,N_discr_center,N_discr_edge):

    # Setting the margins interval #
    w_max=κ                            # Margin max
    w_int_m=max(κ-7*np.sqrt(1-m*m),κ/2)   # Lower value for the extra discretization around w=κ  
    ########
    
    ######## Discretization in two pieces wϵ[0,κ-ε] // wϵ[κ-ε,κ] 
    eps1=w_int_m/(2*N_discr_center)
    eps2=(κ-w_int_m)/(N_discr_edge)
    P_w_cent=np.linspace(eps1        ,w_int_m,N_discr_center)
    P_w_edge=np.linspace(w_int_m+eps2,w_max  ,N_discr_edge  )
    
    P_w_=np.append(P_w_cent,P_w_edge)
    P_w_κ=np.append(-P_w_[::-1],P_w_)
    Δ_w_κ=np.append((np.roll(P_w_κ,-1)-P_w_κ)[0:len(P_w_κ)-1],P_w_κ[1]-P_w_κ[0])
    ########

    def Matrix_creation(P_w,Δ_w):
        ind_max =len(P_w)-1
    
    
        print("Mat (loss): original")
        @numba.jit(nopython=True, fastmath=True)
        def Mat0_simple():

            ##### Tranfer_1(w_new,w_old) = int_(w_old_)^(w_old) dw transfer(w_new,w)  !!! allows to go to smaller values of 1-m    
            transfer1 = lambda w_new,w_old,w_old_: (1/(2*m))*(math.erf(-(w_new-m*w_old)/np.sqrt(2*(1-m*m)))  +  math.erf((w_new-m*w_old_)/np.sqrt(2*(1-m*m))))
            transfer1_= lambda w_new,w_old : np.exp(-(w_new-m*w_old)*(w_new-m*w_old)/(2*(1-m*m)))/np.sqrt(2*np.pi*(1-m*m))

            Mat0=np.zeros((len(P_w),len(P_w)))
            for i in range(len(P_w)):
                for j in range(len(P_w)-1):
                    Mat0[i,j] = transfer1(P_w[i],P_w[j+1],P_w[j])

                Mat0[i,ind_max] = transfer1_(P_w[i],P_w[ind_max])*Δ_w[ind_max]
            return Mat0
            
        Mat0=Mat0_simple()


        print("Mat (loss): adding the fact that P_w varies")
        @numba.jit(nopython=True, fastmath=True)
        def Mat_0_Pw_var(Mat0):
            Mat0_roll_=np.zeros((len(P_w),len(P_w)))
            Mat0_     =np.zeros((len(P_w),len(P_w)))

            for i in range(len(P_w)):
                for j in range(len(P_w)-1):
                    Mat0_roll_[i,j+1]=Mat0[i,j]

            for i in range(len(P_w)):
                for j in range(len(P_w)-1):
                    Mat0_[i,j]=Mat0[i,j]

            return Mat0_roll_/2-Mat0_/2
            
        Mat0_diff_Pw=Mat_0_Pw_var(Mat0)
        Mat0+=Mat0_diff_Pw

        return Mat0
    
    Mat_loss=Matrix_creation(P_w_κ,Δ_w_κ)
    eigenvalue, eigenvector=diag_mat(Mat_loss,N_eigvec=1)
    Loss=abs(eigenvector[:,0   ])/max(2*abs(eigenvector[:,0   ]))\
        +abs(eigenvector[::-1,0])/max(2*abs(eigenvector[::-1,0]))
    

    return eigenvalue,Loss,P_w_κ

def Loc_entropies(κ,m):
    ƛ,Loss_w_κ,P_w_κ=Generating_clique_matrix(κ,m,N_discr_center=4000,N_discr_edge=250)
    
    eps=10**-4
    w_min=0
    w_int=κ-5*np.sqrt(1-m*m)
    w_max=κ
    
    P_w   =np.append(np.linspace(w_min,w_int-eps,80),np.linspace(w_int,w_max,50))
    Loss_w=-np.log(np.interp(P_w,P_w_κ,Loss_w_κ))
    print('End of evaluation')
    return lambda w:np.interp(np.abs(w),P_w,Loss_w)




def MC_Annealed(N,alpha,ticket):
    bias=100
    bias2=0.1
    
    write_='Yes'
    plot_ ='No'
    Ensemble='Macro'
    
    ### No memory correlation ###
    #############################
    
    N_size_no_mem=100000000
    N_no_mem   =2000
    C_no_mem   =np.zeros(N_no_mem)        
    time_no_mem=np.zeros(N_no_mem)  
    t_max=5
    for j in range(N_no_mem):
        time_no_mem[j]=int(j*t_max)/N_no_mem
        C_no_mem[j]=(1-2/N_size_no_mem)**(time_no_mem[j]*N_size_no_mem)
        
    #############################
    #############################
    
    
    
    
    ######## Parameters #########
    #############################
    
    k = np.sqrt(2.6*np.log(N)) # Starting kappa
    k_write= 0                 # Kappa accumulation for writing
    dk = 0.01                  # Kappa step
    P = int(alpha*N)           # Number of patterns

    #############################
    #############################





    ######### Writing ###########
    #############################
    
    if write_=='Yes':
        dk_write=0.01
        
        file1=open('time_esc_'+str(ticket)+'(alpha='+str(alpha)+' N='+str(N)+').txt','w')
        file1.write('kappa   t_esc   t_reject')
        file1.write("\n")
        
        file2=open('Correlation_'+str(ticket)+'(alpha='+str(alpha)+' N='+str(N)+').txt','w')
        file2.write('kappa   t_(C_t_0)   t_rej1(C_t_0)   t_rej2(C_t_0)   C_t_0')
        file2.write("\n")
        
        file3=open('Margins_'+str(ticket)+'(alpha='+str(alpha)+' N='+str(N)+').txt','w')
        file3.write('kappa   margins')
        file3.write("\n")

    #############################
    #############################







    print('Initialization')
    ## Patterns creation
    xis = np.random.randn(P,N)/np.sqrt(N)

    ## Configuration creation
    x = np.random.choice([-1,1],N)

    ## Margins and loss creation
    omegas = np.dot(xis,x)                               # Initial weights
    if Ensemble=='Macro2':
        loss = perceptron_cost_adapt_bias2(omegas,k,bias,bias2) # Initial loss for the perceptron
        
    if Ensemble=='Macro':
        loss = perceptron_cost_adapt_bias(omegas,k,bias)        # Initial loss for the perceptron

        
        
        
    
    t_esc_list=[]
    t_reject1_list=[]
    t_reject2_list=[]
    k_list=[]
    
    time_thres=1500
    Overlap_thres=0.05

    
    
    running = True
    while running:
        if k<3.1:
            dk = 0.003
        if k<1.7:
            dk = 0.0002
            Overlap_thres=0.05

            
        k       -= dk
        k_write += dk
        
        
        t=0  
        t_reject1=0
        t_reject2=0
        x0 = np.copy(x)
        Overlap=1
        
        N_probe=500
        C_probe=np.linspace(1,1.0001*Overlap_thres,N_probe)
        t_probe=np.zeros(N_probe)
        t_probe_rej1=np.zeros(N_probe)
        t_probe_rej2=np.zeros(N_probe)
        index_probe=0
        
        

                
            
        while Overlap > Overlap_thres:
            t+=1
            
            if t/N>time_thres:
                running=False
                break
            
                
            i = np.random.randint(N)                                     # Select a random spin to flip
            omegas_trial = d_interaction_vector_func(omegas,xis,i,-x[i]) # Update the margins 
            
            
            if np.max(np.abs(omegas_trial)) < k:
                
                
                if Ensemble=='Macro':
                    loss_trial = perceptron_cost_adapt_bias(omegas_trial,k,bias)
                if Ensemble=='Macro2':
                    loss_trial = perceptron_cost_adapt_bias2(omegas_trial,k,bias,bias2)

                
                
                a=np.exp(loss-loss_trial)
                b=np.random.rand()
                if a >= b: 
                    print('N=',N)
                    print('alpha=',round(alpha,5),'kappa=',round(k,5),'   t/N=',round(t/N,2),'   x0.x/N = ',round(Overlap,3))
                    print("time rej1:",round(t_reject1/t,3),"time rej2:",round(t_reject2/t,3))
                    print("")
                    omegas = omegas_trial
                    loss = loss_trial
                    x[i] *= -1.0
                    Overlap=np.dot(x0,x)/N
                    
                    if Overlap<C_probe[index_probe]:
                        t_probe[index_probe]=t/N
                        t_probe_rej1[index_probe]=t_reject1/N
                        t_probe_rej2[index_probe]=t_reject2/N
                        index_probe+=1
                        
                else:
                    t_reject2+=1
                    
            else:
                t_reject1+=1
                
        
        if running==True:
            
            k_list=np.append(k_list,[k])
            t_esc_list   = np.append(t_esc_list   ,[t/N]       )
            t_reject1_list= np.append(t_reject1_list,[t_reject1/N])
            t_reject2_list= np.append(t_reject2_list,[t_reject2/N])
        
        
        if write_=='Yes':
            if k_write>dk_write:
                file2.write(str(k))
                file2.write('		')
                for l in range(N_probe):
                    file2.write(str(t_probe[l]))
                    file2.write('		')
                    
                for l in range(N_probe):
                    file2.write(str(t_probe_rej1[l]))
                    file2.write('		')
                    
                for l in range(N_probe):
                    file2.write(str(t_probe_rej2[l]))
                    file2.write('		')
                    
                    
                for l in range(N_probe):
                    file2.write(str(C_probe[l]))
                    file2.write('		')
                file2.write("\n")
                
                
                
                
                file3.write(str(k))
                file3.write('		')
                for l in range(P):
                    file3.write(str(omegas[l]))
                    file3.write('		')
                file3.write("\n")
                    
              
        if plot_=='Yes':
            
            def Matching(time_no_mem,C_no_mem,time_,C_):

                C_min=max(min(C_),0.3)
                        
                N_sample=15
                C_sample=np.linspace(C_min,1,N_sample)
                time_sample_no_mem=np.zeros(N_sample)
                time_sample       =np.zeros(N_sample)
                        
                        
                for t in range(N_sample):
                    ind_sample_no_mem    =np.argmin(abs(C_no_mem-C_sample[t]))
                    time_sample_no_mem[t]=time_no_mem[int(ind_sample_no_mem)]
                            
                    ind_sample           =np.argmin(abs(C_      -C_sample[t]))
                    time_sample       [t]=time_      [int(ind_sample)]

                def Find_rate(ɣ):
                    return np.sum(abs(time_sample/ɣ-time_sample_no_mem))
                        
            
                return optimize.fmin(Find_rate,10,disp=False)            
                        
                        
            rate=Matching(time_no_mem,C_no_mem,t_probe,C_probe) 
            

            plt.plot(t_probe/rate,C_probe,label=str(round(rate[0],4)))   
            plt.plot(time_no_mem ,C_no_mem,linestyle='--')  
            plt.xlabel(r'$t/N$')
            plt.ylabel(r'${\bf x}_t \cdot {\bf x}_0 /N$')
            plt.legend()
            plt.show()             
        
            plt.hist(omegas,bins='sqrt',density=True)
            


            omegas_th = np.linspace(-k,k,200)
            distrib_1 = np.zeros(len(omegas_th))
            distrib_2 = np.zeros(len(omegas_th))
            distrib_3 = np.zeros(len(omegas_th))
            for l in range(len(omegas_th)):
                distrib_1[l]=np.exp(-omegas_th[l]*omegas_th[l]/2-perceptron_cost_adapt_bias (np.array([omegas_th[l]]),k,bias))
                distrib_2[l]=np.exp(-omegas_th[l]*omegas_th[l]/2-perceptron_cost_adapt_bias2(np.array([omegas_th[l]]),k,bias,bias2))
                distrib_3[l]=np.exp(-omegas_th[l]*omegas_th[l]/2)
                
            Norm_1=integrate.quad(lambda w:np.interp(w,omegas_th,distrib_1),-k,k)[0]
            Norm_2=integrate.quad(lambda w:np.interp(w,omegas_th,distrib_2),-k,k)[0]
            Norm_3=integrate.quad(lambda w:np.interp(w,omegas_th,distrib_3),-k,k)[0]

            
            distr_th1=distrib_1/Norm_1
            distr_th2=distrib_2/Norm_2
            distr_th3=distrib_3/Norm_3
            
            plt.plot(omegas_th,distr_th1)
            plt.plot(omegas_th,distr_th2)
            plt.plot(omegas_th,distr_th3,linestyle='--')
            plt.ylim([0,1.3*max(distr_th1)])
            plt.xlabel(r'$w=\xi\cdot{\bf x}$')
            plt.ylabel(r'$P(w)$')
            plt.show()
            
            
            
            
                    
        print('Annealing in kappa, k = ',k)
        print('')        
        print('')        
        print('')
        print('')        
        print('')        
        print('')
        
        
    if write_=='Yes':
        for l in range(len(k_list)):
            file1.write(str(k_list[l]))
            file1.write('		')
            file1.write(str(t_esc_list[l]))
            file1.write('		')
            file1.write(str(t_reject1_list[l]))
            file1.write('		')
            file1.write(str(t_reject2_list[l]))
            file1.write("\n")
        

    if plot_=='Yes':
        plt.plot(k_list,t_esc_list)
        plt.ylabel(r'$t_{\rm esc.}/N$')
        plt.xlabel(r'$\kappa$')
        plt.show()
        
        plt.plot(k_list,t_esc_list-t_reject2_list-t_reject1_list)
        plt.ylabel(r'$t_{\rm esc.}/N-t_{\rm rej.}/N$')
        plt.xlabel(r'$\kappa$')
        plt.show()
        
        plt.plot(k_list,t_reject1_list/t_esc_list,label='reject wrong margin')
        plt.plot(k_list,t_reject2_list/t_esc_list,label='reject energ gap')
        plt.plot(k_list,t_reject2_list/t_esc_list+t_reject1_list/t_esc_list,label='reject total')
        plt.legend()
        plt.xlabel(r'$\kappa$')
        plt.show()
        
        
        
alpha_list=[0.75]
ticket=-1
N_list=[5000]
for k in range(len(N_list)):
    for j in range(len(alpha_list)):
        MC_Annealed(N_list[k],alpha_list[j],ticket)  
    
    
    
    
    
    
    