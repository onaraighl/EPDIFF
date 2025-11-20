function [x,y,X,Y,u_hist,v_hist,rhobar_hist,t_vec,energy_vec,EC1_vec,EC2_vec,k_bins,E_k] = CH2D()
% function [x,y,X,Y,u,v,C] = CH2D()

% *************************************************************************
% 2DCH / EPDiff pseudospectral code.
% October 2025
% Lennon Ó Náraigh (and ChatGPT)
% 
% Inputs: null
% Outputs: the mesh [X,Y] and the velocity components u and v at the final
% time T.
%
% KEY POINTS:
%
% * Final time T is controlled below under "numerical parameters".
% * Intermediate u and v values are printed to the screen every snapEvery
%   timesteps, where snapEvery is controlled below also under "numerical
%   parameters".

% *************************************************************************
% Technical:
%
% TEMPORAL DISCRETIZATION:
% RK4, hence EXPLICIT
% 
% Note: On 07/11/2025 I tried AB3 as an alternative but it has the same
% stabiilty propoerties as RK4.
%
% SPATIAL DERIVATIVES:
% Computed using Fourier transforms, hence psuedospectral method.
%
% REFERENCES:
% 1.    Holm, D.D. and Staley, M.F., 2013. Interaction dynamics of singular 
%       wave fronts. arXiv preprint arXiv:1301.1460.
% 2.    Chertock, A., Du Toit, P. and Marsden, J.E., 2012. 
%       Integration of the EPDiff equation by particle methods.
%       ESAIM: Mathematical Modelling and Numerical Analysis, 46(3), pp.515-534.
% 3.    Azencot, O., Vantzos, O. and Ben‐Chen, M., 2018, August. An explicit 
%       structure‐preserving numerical scheme for EPDiff. 
%       In Computer Graphics Forum (Vol. 37, No. 5, pp. 107-119).

% *************************************************************************
% Physical parameters: alpha=alpha2=0.5 for the videos.

alpha=0.5;
alphasq=alpha^2;

alpha2=alpha/2;
alphasq2=alpha2^2;

g=1; % only change this if you really know what's going on.

nup=0;

% *************************************************************************
% Numerical parameters

Lx=12*pi;
Ly=12*pi;

Nx=512;
Ny=512;

dx = Lx/Nx; % grid spacing
dy = Ly/Ny; % grid spacing

T=10; % final time
dt=1e-3; 

Nt=ceil(T/dt);

t_val=0; % time value, initially at zero.
snapEvery=100; % snapshot plotted every snapEvery timesteps
snapI=0; % snapshot counter

% vector of energy values
t_vec=0*(1:Nt);
energy_vec=0*(1:Nt);
EC1_vec=0*(1:Nt);
EC2_vec=0*(1:Nt);

% *************************************************************************
% Numerical preliminaries:

x = (0:Nx-1)*dx;
y = (0:Ny-1)*dy;

% Centred coordinate system: 
x=x-(Lx/2);
y=y-(Ly/2);

% Create mesh in real space:
[X,Y] = meshgrid(x,y); 

% Wavvenumber grids (from ChatGPT):
kx = (2*pi/Lx) * [0:(Nx/2-1), -Nx/2:-1];
ky = (2*pi/Ly) * [0:(Ny/2-1), -Ny/2:-1];
[KX, KY] = meshgrid(kx, ky);   % note: meshgrid gives (rows = y), consistent with fft2 ordering

% Other operators:
K2 = KX.^2 + KY.^2;
K4=K2.^2;
Darryl_op=(1+alphasq*K2).^(-1);
Darryl_op2=(1+alphasq2*K2).^(-1);
% Darryl_op_sigma=(1+(sigma^2)*K2).^(-1);

% Use standard de-aliasing, with 2/3 rule cutoff
kx_cut = (2*pi/Lx)*(2/3) * (Nx/2);
ky_cut = (2*pi/Ly)*(2/3) * (Ny/2);

% Dealiasing mask
mask = (abs(KX) < kx_cut) & (abs(KY) < ky_cut);

% *************************************************************************
% Initial conditions
% *************************************************************************

% % % % Option 1: ONe breaking
% % 
wx=2;
wy=2;

xc1=0;

rhobar=1+(tanh((X-xc1)+wx)-tanh((X-xc1)-wx)).*(tanh(Y+wy)-tanh(Y-wy));

% % Option 2: Two dams breaking

% % wx=2;
% % wy=2;
% % 
% % xc1=-6;
% % xc2= 6;
% % 
% % rhobar1=1+(tanh((X-xc1)+wx)-tanh((X-xc1)-wx)).*(tanh(Y+wy)-tanh(Y-wy));
% % rhobar2=1+(tanh((X-xc2)+wx)-tanh((X-xc2)-wx)).*(tanh(Y+wy)-tanh(Y-wy));
% % 
% % rhobar=(rhobar1+rhobar2)/2;

% % % Option 3: Gaussian hump
% wx=1;
% Rsq=X.^2+Y.^2;
% rhobar=1+exp(-Rsq/(2*wx^2));

% % % Option 4: taylor-green type initial condition
% % With Miguel on 04/11
% 
% u_init=cos( (2*pi/Lx)*X).*cos((2*pi/Ly)*Y)+1;
% v_init=sin( (2*pi/Lx)*X).*cos((2*pi/Ly)*Y);
% % rhobar=2+0*X;
% rhobar=1+0*X;


% *************************************************************************
% Finalize initial conditions:

u_init=0*X;
v_init=0*X;

u = u_init;
v = v_init;

% *************************************************************************
% Auxiliary variable:

varphi_init=0*X;
varphi=varphi_init;

% *************************************************************************
% *************************************************************************
% Move arrays to GPU (ChatGPT):

KX = gpuArray(KX);
KY = gpuArray(KY);
u = gpuArray(u);
v = gpuArray(v);
varphi=gpuArray(varphi);
% C = gpuArray(C);

K2 = gpuArray(K2);
Darryl_op=gpuArray(Darryl_op);
Darryl_op2=gpuArray(Darryl_op2);
mask=gpuArray(mask);

% *************************************************************************

uhat=fft2(u); 
vhat=fft2(v);
rhobar_hat=fft2(rhobar);
varphi_hat=fft2(varphi);

for t_ctr=1:Nt

    % RK4 update:
    [k1u, k1v,k1_rhobar,k1_varphi] = RHS_RK4(uhat,              vhat             ,rhobar_hat                 ,varphi_hat);
    [k2u, k2v,k2_rhobar,k2_varphi] = RHS_RK4(uhat + 0.5*dt*k1u, vhat + 0.5*dt*k1v,rhobar_hat+0.5*dt*k1_rhobar,varphi_hat+0.5*dt*k1_varphi);
    [k3u, k3v,k3_rhobar,k3_varphi] = RHS_RK4(uhat + 0.5*dt*k2u, vhat + 0.5*dt*k2v,rhobar_hat+0.5*dt*k2_rhobar,varphi_hat+0.5*dt*k2_varphi);
    [k4u, k4v,k4_rhobar,k4_varphi] = RHS_RK4(uhat +     dt*k3u, vhat +     dt*k3v,rhobar_hat+    dt*k3_rhobar,varphi_hat+    dt*k3_varphi);
    
    % update Fourier coefficients
    uhat = (uhat.*(1-(nup*dt/2)*K4) + (dt/6)*(k1u + 2*k2u + 2*k3u + k4u))./(1+(nup*dt/2)*K4);
    vhat = (vhat.*(1-(nup*dt/2)*K4) + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v))./(1+(nup*dt/2)*K4);
    rhobar_hat=rhobar_hat+(dt/6)*(k1_rhobar+2*k2_rhobar+2*k3_rhobar+k4_rhobar);
    varphi_hat = varphi_hat + (dt/6)*(k1_varphi + 2*k2_varphi + 2*k3_varphi + k4_varphi);

    % transfor back to real space for visualization
    u=real(ifft2(uhat)); v=real(ifft2(vhat)); rhobar=real(ifft2(rhobar_hat)); varphi=real(ifft2(varphi_hat));
    varphi=varphi-sum(sum(varphi))/(Nx*Ny); % subtract off average

    lap_rhobar = -real(ifft2(K2.*rhobar_hat));
    rho=rhobar-alphasq*lap_rhobar;

    rho=max(rho,1e-8); % clipping

    % keep track of time - literally!
    t_val=t_val+dt;

    %**********************************************************************
    % Analysis:
    energy_t=get_energy(K2,rhobar_hat,uhat,vhat,alphasq,g,dx,dy); % energy

    [C1,C1_sq_times_rho]=get_C1(KX,KY,K2,rhobar_hat,uhat,vhat,alphasq); % potential vorticity q1
    EC1_t=sum(sum(C1_sq_times_rho))*dx*dy;

    [C2,C2_sq_times_rho]=get_C2(KX,KY,K2,rhobar_hat,uhat,vhat,varphi_hat,alphasq); % conserved quantiity q2
    EC2_t=sum(sum(C2_sq_times_rho))*dx*dy;

    % Note q1 and q2 will only make sense if h is non-zero.

    t_vec(t_ctr)=t_val;
    energy_vec(t_ctr)=energy_t;
    EC1_vec(t_ctr)=EC1_t;
    EC2_vec(t_ctr)=EC2_t;

    %**********************************************************************

    % Plot and monitor output:
    if mod(t_ctr,snapEvery)==0

        snapI=snapI+1;

        % Bring results back to CPU
        u_cpu = gather(u);
        v_cpu = gather(v);
        rho_cpu=gather(rho);
        rhobar_cpu=gather(rhobar);
        varphi_cpu=gather(varphi);
        C1_cpu=gather(C1);
        C2_cpu=gather(C2);

        % gather results to space-time array.
        u_hist(snapI,:,:)=u_cpu; v_hist(snapI,:,:)=v_cpu;  rhobar_hist(snapI,:,:)=rhobar_cpu;

        rhobar_hat=gather(rhobar_hat);
        uhat=gather(uhat);
        vhat=gather(vhat);
        [k_bins,E_k] = get_energy_spectrum(KX,KY,K2,rhobar_hat,uhat,vhat,Nx,Ny,Lx,alphasq,g);
        k_bins = gather(k_bins);

        % output results to screen
        clf;
        subplot(1,3,1); imagesc(x,y,sqrt(u_cpu.^2+v_cpu.^2)); axis xy equal tight;
        title(strcat('|u|, t=',num2str(t_val))); colorbar;
        xlabel('x')
        ylabel('y')
        subplot(1,3,2); imagesc(x,y,rhobar_cpu); axis xy equal tight;
        title('rho');  colorbar;
        xlabel('x')
        ylabel('y')
        subplot(1,3,3); imagesc(x,y,C2); axis xy equal tight;
        title('C2'); colorbar;
        xlabel('x')
        ylabel('y')
        % subplot(1,3,3); loglog(k_bins,E_k,'-o')
        % xlabel('k')
        % ylabel('E(k)')
        drawnow;

        min(rho_cpu(:))
    end

end

% % Bring results back to CPU
% u = gather(u);
% v = gather(v);
% C = gather(C);

rhobar_hat=gather(rhobar_hat);
uhat=gather(uhat);
vhat=gather(vhat);
[k_bins,E_k] = get_energy_spectrum(KX,KY,K2,rhobar_hat,uhat,vhat,Nx,Ny,Lx,alphasq,g);
k_bins = gather(k_bins);


% *************************************************************************
% RHS calculation for RK4 
% "L" for local in each of the variable names.

function [RHS_x_hat, RHS_y_hat,RHS_rhobar_hat,RHS_varphi_hat] = RHS_RK4(u_hatL,v_hatL,rhobar_hatL,varphi_hatL) 
    % transform to physical space:
    uL = real(ifft2(u_hatL)); 
    vL = real(ifft2(v_hatL));
    rhobarL=real(ifft2(rhobar_hatL));

    % spectral derivatives
    u_xL  =  real(ifft2(1i*KX .*u_hatL));
    u_yL  =  real(ifft2(1i*KY .*u_hatL));
    lap_uL = -real(ifft2(K2.*u_hatL));

    v_xL  =  real(ifft2(1i*KX .*v_hatL));
    v_yL  =  real(ifft2(1i*KY .*v_hatL));
    lap_vL = -real(ifft2(K2.*v_hatL));

    rhobar_xL  =  real(ifft2(1i*KX .*rhobar_hatL));
    rhobar_yL  =  real(ifft2(1i*KY .*rhobar_hatL));
    lap_rhobarL = -real(ifft2(K2.*rhobar_hatL));

    % bare density
    rhoL=rhobarL-alphasq2*lap_rhobarL;

    % divergence:
    divu=u_xL+v_yL;
    
    % convective derivatives:
    mxL=uL-alphasq*lap_uL;
    myL=vL-alphasq*lap_vL;

    mx_hat=fft2(mxL);
    dx_mx=real(ifft2(1i*KX.*mx_hat));
    dy_mx=real(ifft2(1i*KY.*mx_hat));

    my_hat=fft2(myL);
    dx_my=real(ifft2(1i*KX.*my_hat));
    dy_my=real(ifft2(1i*KY.*my_hat));

    conv_x=uL.*dx_mx+vL.*dy_mx;
    conv_y=uL.*dx_my+vL.*dy_my;

    % Note - this contraction was wrong.
    % stretch_x=mx.*u_xL + my.*u_yL;
    % stretch_y=mx.*v_xL + my.*v_yL;

    stretch_x=mxL.*u_xL + myL.*v_xL;
    stretch_y=mxL.*u_yL + myL.*v_yL;

    b_param=2;

    expand_x=mxL.*divu*(b_param-1);
    expand_y=myL.*divu*(b_param-1);

    % buoyancy terms
    buoy_x=g*rhobar_xL.*rhoL;
    buoy_y=g*rhobar_yL.*rhoL;

    RHS_x=-conv_x-stretch_x-expand_x-buoy_x;
    RHS_y=-conv_y-stretch_y-expand_y-buoy_y;

    % *********************************************************************
    % Construct final RHS in Fourier space
    RHS_x_hat = fft2(RHS_x);
    RHS_y_hat = fft2(RHS_y);

    % Apply filter
    RHS_x_hat = mask .* RHS_x_hat;
    RHS_y_hat = mask .* RHS_y_hat;

    % Apply inverse Helmholtz
    RHS_x_hat = Darryl_op .* RHS_x_hat;
    RHS_y_hat = Darryl_op .* RHS_y_hat;

    % *********************************************************************
    % same for density equation
    
    temp_x=uL.*rhoL;
    temp_y=vL.*rhoL;

    temp_x_hat=fft2(temp_x);
    temp_y_hat=fft2(temp_y);

    dx_temp_x_hat=1i*KX.*temp_x_hat;
    dy_temp_y_hat=1i*KY.*temp_y_hat;

    % Apply filter and inverse Helmholtz:
    RHS_rhobar_hat=-Darryl_op2.*mask.*(dx_temp_x_hat+dy_temp_y_hat);

    % *********************************************************************
    % same for varphi-equation:


    dx_varphi_hat=1i*KX.*varphi_hatL;
    dy_varphi_hat=1i*KY.*varphi_hatL;

    dx_varphi=real(ifft2(dx_varphi_hat));
    dy_varphi=real(ifft2(dy_varphi_hat));

    conv=uL.*dx_varphi+vL.*dy_varphi;
    conv_hat=fft2(conv);
    RHS_varphi_hat=-mask.*conv_hat+g*rhobar_hatL;

end

% *************************************************************************

end






