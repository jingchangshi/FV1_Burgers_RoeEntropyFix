import numpy as np

def Flux(in_u_arr):
    # Burgers equation
    f_arr=0.5*in_u_arr**2
    return f_arr

def IC(in_x,in_type):
    if(in_type==1):
        u_arr=np.sin(in_x)+1.5
    elif(in_type==2):
        u_arr=np.sign(in_x-0.25*2*np.pi)*np.sign(0.75*2*np.pi-in_x)
        idx_pos=u_arr>0
        u_arr[idx_pos]*=2
    else:
        exit("Not implemented!")
    return u_arr

def RK1(in_u_arr,in_dt,in_dx):
    dF_arr=Res(in_u_arr,in_dx)
    u_new_arr=in_u_arr-in_dt*dF_arr
    return u_new_arr

def Res(in_u_arr,in_dx):
    # Periodic BC
    u_prev_arr=np.append(in_u_arr[-1],in_u_arr[:-1])
    u_next_arr=np.append(in_u_arr[1:],in_u_arr[0])
    #  F_l_arr=RoeFlux(u_prev_arr,in_u_arr)
    #  F_r_arr=RoeFlux(in_u_arr,u_next_arr)
    F_l_arr=RoeFluxEntropyFix(u_prev_arr,in_u_arr)
    F_r_arr=RoeFluxEntropyFix(in_u_arr,u_next_arr)
    dF_arr=F_r_arr-F_l_arr
    dF_arr/=in_dx
    return dF_arr

def RoeFlux(in_u_l_arr,in_u_r_arr):
    a_arr=0.5*(in_u_l_arr+in_u_r_arr)
    f_l_arr=Flux(in_u_l_arr)
    f_r_arr=Flux(in_u_r_arr)
    f_arr=0.5*(f_l_arr+f_r_arr)-0.5*np.abs(a_arr)*(in_u_r_arr-in_u_l_arr)
    return f_arr

def RoeFluxEntropyFix(in_u_l_arr,in_u_r_arr):
    a_arr=0.5*(in_u_l_arr+in_u_r_arr)
    a_l_arr=in_u_l_arr
    a_r_arr=in_u_r_arr
    epsilon_arr=np.maximum(a_arr-a_l_arr,a_r_arr-a_arr)
    epsilon_arr=np.maximum(np.zeros(epsilon_arr.shape),epsilon_arr)
    idx_pos=a_arr>=epsilon_arr
    idx_neg=a_arr<epsilon_arr
    a_arr[idx_pos]=np.abs(a_arr[idx_pos])
    #  a_arr[idx_neg]=0.5*(a_arr[idx_neg]**2/epsilon_arr[idx_neg]+epsilon_arr[idx_neg])
    a_arr[idx_neg]=epsilon_arr[idx_neg]
    a_mod_arr=a_arr
    f_l_arr=Flux(in_u_l_arr)
    f_r_arr=Flux(in_u_r_arr)
    f_arr=0.5*(f_l_arr+f_r_arr)-0.5*a_mod_arr*(in_u_r_arr-in_u_l_arr)
    return f_arr

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.style.use('sjc')
    x_l_edge=0.0
    x_h_edge=1.0
    x_h_edge*=2*np.pi
    n_x_edge=200
    n_x_edge+=1
    x_edge_arr=np.linspace(x_l_edge,x_h_edge,n_x_edge)
    x_cell_arr=(x_edge_arr[0:-1]+x_edge_arr[1:])/2.0
    IC_type=2
    u_cell_arr=IC(x_cell_arr,IC_type)
    dx=x_edge_arr[1]-x_edge_arr[0]
    cfl=0.8
    n_iter=100000
    end_time=0.5
    time=0
    i_iter=0
    plot_progress=False
    #  plot_progress=True
    while(time<end_time and i_iter<n_iter):
        if(plot_progress):
            fig=plt.figure()
            ax=fig.gca()
            ax.plot(x_cell_arr,u_cell_arr,'o-')
            ax.set_xlabel("X")
            ax.set_ylabel("U")
            ax.set_title("it=%d,t=%.3f"%(i_iter,time))
            plt.show()
            plt.close(fig)
        i_iter+=1
        dt=cfl*dx/np.max(np.abs(u_cell_arr))
        if(time+dt>end_time):
            dt=end_time-time
        time+=dt
        u_cell_arr=RK1(u_cell_arr,dt,dx)
    fig=plt.figure()
    ax=fig.gca()
    ax.plot(x_cell_arr,u_cell_arr,'o-')
    ax.set_xlabel("X")
    ax.set_ylabel("U")
    fig_name="U_it%d_t%.3f.png"%(i_iter,time)
    plt.savefig(fig_name)
