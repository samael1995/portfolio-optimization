clc
clear
close all
%% loading the stock data
T = readtable('stock.csv');
data = T(:,2:end).Variables;
stock_name = ["INFY.NS", "TCS.NS","TATAMOTORS.NS","MARUTI.NS",...
	"SUNPHARMA.NS","CIPLA.NS","ITC.NS","MARICO.NS","GOLDBEES.NS"];
%% data prep
result = zeros(4,5);
r = (data(1:end-1,:) - data(2:end,:))./data(2:end,:);
[T, n] = size(r);
coskewness=zeros(n,n^2);
cokurtosis=zeros(n,n^3);
mu = mean(r);
r_demeaned = (r - kron(mu, ones(T,1)));
varcov = 1/length(r)*(r_demeaned)'*(r_demeaned);

for i=1:T
    coskewness=coskewness+1/T*(kron(r_demeaned(i,:)'*r_demeaned(i,:),r_demeaned(i,:)));
    cokurtosis=cokurtosis+1/T*(kron(r_demeaned(i,:)',kron(r_demeaned(i,:),kron(r_demeaned(i,:),r_demeaned(i,:)))));
end

%% model optimization based on  polynomial goal programming (PGP) algorithm
%% first stage: finding R*, V*, S*, K* and E* sepreately
w0 = ones(n,1)/n;
A = [];
b = [];
Aeq = ones(1,n);
beq = 1;
ub = ones(1,n);
lb = zeros(1,n);

% R: maximize (W'mu) , sum(W)=1 , Wi>0
fun1 =  @(w)-1*w'*mu';
W_R = fmincon(fun1,w0,A,b,Aeq,beq,lb,ub);
opt_R = mu*W_R;

% V: minimize(W'*V*W), sum(W)=1 , Wi>0
fun2 = @(w)w'*varcov*w;
w_var = fmincon(fun2,w0,A,b,Aeq,beq,lb,ub);
opt_var = w_var'*varcov*w_var;

% S: maximize w'*coskewness*kron(w,w)
fun3 = @(w)-1*w'*coskewness*kron(w,w)/(w'*varcov*w)^(3/2);
w_coskewness = fmincon(fun3,w0,A,b,Aeq,beq,lb,ub);
opt_skewness = w_coskewness'*coskewness*kron(w_coskewness,w_coskewness)/(w_coskewness'*varcov*w_coskewness)^(3/2);

% K: minimize w'*cokurtosis*kron(kron(w,w),w), sum(W)=1 , Wi>0
fun4 = @(w)w'*cokurtosis*kron(kron(w,w),w)/(w'*varcov*w)^2;
w_kurtosis = fmincon(fun4,w0,A,b,Aeq,beq,lb,ub);
opt_kurtosis = w_kurtosis'*cokurtosis*kron(kron(w_kurtosis,w_kurtosis),w_kurtosis)/(w_kurtosis'*varcov*w_kurtosis)^2;

% E: maximize(-W'*log(W)), sum(W)=1 , Wi>0
fun5 = @(w)-1*-w'*log(w);
w_entropy = fmincon(fun5,w0,A,b,Aeq,beq,lb,ub);
opt_entropy = -w_entropy'*log(w_entropy)/log(n);


%% second stage : minimzing Z function
% calculate weights for MVSK model
L = [1,1,1,1,0];
final_fun = @(w)(1+ abs((opt_R-w'*mu')/(opt_R)))^L(1)+...
    +(1+ abs((w'*varcov*w + opt_var)/(opt_var)))^L(2)...
    +(1+ abs((w'*coskewness*kron(w,w)/(w'*varcov*w)^(3/2)-opt_skewness)/(opt_skewness)))^L(3)...
    +(1+ abs((w'*cokurtosis*kron(kron(w,w),w)/(w'*varcov*w)^2+opt_kurtosis)/(opt_kurtosis)))^L(4)...
    +(1+ abs((w'*log(w)/log(n) + opt_entropy)/(opt_entropy)))^L(5);
w = fmincon(final_fun,w0,A,b,Aeq,beq,lb,ub);

result(1,:) = [opt_R,opt_var, opt_skewness, opt_kurtosis,opt_entropy];

result(2,:)= [w0'*mu', w0'*varcov*w0 , w0'*coskewness*kron(w0,w0)/(w0'*varcov*w0)^(3/2),...
    w0'*cokurtosis*kron(kron(w0,w0),w0)/(w0'*varcov*w0)^2, -w0'*log(w0)/log(n) ];
result(3,:) = [w'*mu', w'*varcov*w , w'*coskewness*kron(w,w)/(w'*varcov*w)^(3/2),...
    w'*cokurtosis*kron(kron(w,w),w)/(w'*varcov*w)^2, -w'*log(w)/log(n) ];
% calculate weights for MVSKE model
L = [1,1,1,1,1];
final_fun = @(w)(1+ abs((opt_R-w'*mu')/(opt_R)))^L(1)+...
    +(1+ abs((w'*varcov*w + opt_var)/(opt_var)))^L(2)...
    +(1+ abs((w'*coskewness*kron(w,w)/(w'*varcov*w)^(3/2)-opt_skewness)/(opt_skewness)))^L(3)...
    +(1+ abs((w'*cokurtosis*kron(kron(w,w),w)/(w'*varcov*w)^2+opt_kurtosis)/(opt_kurtosis)))^L(4)...
    +(1+ abs((w'*log(w)/log(n) + opt_entropy)/(opt_entropy)))^L(5);

w_E = fmincon(final_fun,w0,A,b,Aeq,beq,lb,ub);
result(4,:) = [w_E'*mu', w_E'*varcov*w_E , w_E'*coskewness*kron(w_E,w_E)/(w_E'*varcov*w_E)^(3/2),...
    w_E'*cokurtosis*kron(kron(w_E,w_E),w_E)/(w_E'*varcov*w_E)^2, -w_E'*log(w_E)/log(n) ];


Result.w_MVSK = w;
Result.w_MVSKE = w_E;
Result.result = result;
save("Result","Result")

%%  plotting results
subplot(121)
h1 = pie(w);
legend(stock_name)
title("stock share(%) based on MVSK portfolio optimization")
th = findobj(h1,'Type','Text');
isSmall = startsWith({th.String}, '<');
set(th(isSmall),'String', '')

subplot(122)
h2 =pie(w_E);
legend(stock_name)
title("stock share(%) based on MVSKE portfolio optimization") 
th = findobj(h2,'Type','Text');
isSmall = startsWith({th.String}, '<');
set(th(isSmall),'String', '')
%% result table
portfolio = ["optimum_value";"equal_weight(1/n)";"MVSK";"MVSKE"];
mean = result(:,1);
std = result(:,2);
skewness = result(:,3);
kurtosis = result(:,4);
entropy = result(:,5);

result_table = table(portfolio,mean,std,skewness,kurtosis,entropy);
disp(result_table)



