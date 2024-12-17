% MMECG after SST (in the form of figure)
%
% load([int2str(1),'.mat']);
%
% subplot(3,1,1)
% plot(data.RCG(:,4))
% subplot(3,1,2)
% plot(data.RCG(:,6))
% subplot(3,1,3)
% plot(data.RCG(:,8))
c=0;
f_s=200;
% plot_range=3030:3030+600*2;
% plot_range=3030:5030;
for ID=1:91
    fprintf(['---------obj ', int2str(ID), '---------\n']);
    load(['data_org/',int2str(ID),'.mat']);
    ecgSignal=data.ECG;
    RCG=data.RCG;
    SST=[];
    fprintf('SST for Point ');
    for s=1:50
        RCG=data.RCG(:,s);
%         RCG=RCG(plot_range);
%         RCG = (RCG - min(RCG))/(max(RCG)-min(RCG)); % normlization
        % RCG_re=resample(RCG,f_d,f_s);
        fprintf([int2str(s), ', ']); 
        [sst,f] = wsst(RCG,f_s,'VoicesPerOctave',10);
        freq_len=length(sst(:,1));
        SST(s,:,:)=abs(sst(freq_len/2:freq_len,:));     
    end
    f_d=30; %f_d for desired frequency
    fprintf(['\nResample with ', int2str(f_d), 'Hz\n']);
    SST=resample_sst(SST,f_s,f_d);
    for id=1:50
        temp=SST(id,:,:);
        k=1/(max(max(temp))-min(min(temp)));
        SST(id,:,:)= k*(temp-min(min(temp)));  %Normalization [0,1]
%         k=2/(max(max(temp))-min(min(temp)));
%         SST(id,:,:)= -1 +k*(temp-min(min(temp)));  %Normalization[-1,1]
    end

    save(['./data_sst/', int2str(f_d), 'Hz_half_01/SST_obj',int2str(data.id),'_',data.physistatus,'_',int2str(ID),'_', int2str(f_d), 'Hz.mat'],'SST','-v7.3')
end

close all

% resample the sst plot in time axis
function x_sampled=resample_sst(sst,f_org,f_desired)
x_sampled=[];
[~, freq_count, ~]=size(sst);
for i=1:50
    for j=1:freq_count
        x_sampled(i,j,:) = resample(squeeze(sst(i,j,:)),f_desired,f_org);
    end
end
end

function plot3Dpoint(data)
len=size(data.posXYZ);

figure()
for i=1:len(1)
    plot3(0,0,0,'x')
        x=data.posXYZ(:,1);
        y=data.posXYZ(:,2);
        z=data.posXYZ(:,3);
        plot3(x,y,z,'o')
        hold on
end
end

function ecg_smooth=smooth_ECG(ecgSignal,Fs)
% ecgSignal = sgolayfilt(ecgSignal,3,17); %1st
ecgSignal = sgolayfilt(ecgSignal,9,19);

originalFs = Fs;
desiredFs = 1000; % desired Fs for the network
[p,q] = rat(desiredFs / originalFs);
% ecgSignal = resample(ecgSignal,p,q);
% Fs=1000;

%LPF
Fpass  = 10;
Fstop = 40;
Dpass = 0.05;
Dstop = 0.0001;
F     = [0 Fpass Fstop Fs/2]/(Fs/2);
A     = [1 1 0 0];
D     = [Dpass Dstop];
b = firgr('minorder',F,A,D);
LP = dsp.FIRFilter('Numerator',b);
% %HPF
% Fstop = 200;
% Fpass = 400;
% Dstop = 0.0001;
% Dpass = 0.05;
% F = [0 Fstop Fpass Fs/2]/(Fs/2); % Frequency vector
% A = [0 0 1 1]; % Amplitude vector
% D = [Dstop Dpass];   % Deviation (ripple) vector
% b  = firgr('minord',F,A,D);
% HP = dsp.FIRFilter('Numerator',b);

ecg_smooth = LP(ecgSignal);


end

function fakeLabel(ecgSignal,Fs,data,ID,resample)
%         if need resample
if resample
    originalFs = Fs;
    desiredFs = 250; % desired Fs for the network
    [p,q] = rat(desiredFs / originalFs);
    ecgSignal = resample(ecgSignal,p,q);
    Fs=desiredFs;
end

ROILimits = [[1 2];[3 4];[5 6];[7 8]];
label=["P";"QRS";"T"];
B = categorical(label);
Value = [B(1);B(2);B(3);B(1)];
signalRegionLabels = table(ROILimits,Value);
%id->subject id; ID->file ID
save(['./seg/ecgSig_',int2str(data.id),'_',data.physistatus,'_',int2str(ID),'_smooth','.mat'],'ecgSignal','signalRegionLabels','Fs')
end
function test()
x = ecg(500).';
y = sgolayfilt(x,0,5);
[M,N] = size(y);

Fs = 1000;
TS = timescope('SampleRate',Fs,...
    'TimeSpanSource','Property',...
    'TimeSpan',1.5,...
    'ShowGrid',true,...
    'NumInputPorts',2,...
    'LayoutDimensions',[2 1]);
TS.ActiveDisplay = 1;
TS.YLimits = [-1,1];
TS.Title = 'Noisy Signal';
TS.ActiveDisplay = 2;
TS.YLimits = [-1,1];
TS.Title = 'Filtered Signal';

Fpass  = 200;
Fstop = 400;
Dpass = 0.05;
Dstop = 0.0001;
F     = [0 Fpass Fstop Fs/2]/(Fs/2);
A     = [1 1 0 0];
D     = [Dpass Dstop];
b = firgr('minorder',F,A,D);
LP = dsp.FIRFilter('Numerator',b);

Fstop = 200;
Fpass = 400;
Dstop = 0.0001;
Dpass = 0.05;
F = [0 Fstop Fpass Fs/2]/(Fs/2); % Frequency vector
A = [0 0 1 1]; % Amplitude vector
D = [Dstop Dpass];   % Deviation (ripple) vector
b  = firgr('minord',F,A,D);
HP = dsp.FIRFilter('Numerator',b);
tic;
while toc < 30
    x = .1 * randn(M,N);
    highFreqNoise = HP(x);
    noisySignal = y + highFreqNoise;
    filteredSignal = LP(noisySignal);
    TS(noisySignal,filteredSignal);
end

% Finalize
release(TS)
end

function x = ecg(L)
a0 = [0,  1, 40,  1,   0, -34, 118, -99,   0,   2,  21,   2,   0,   0,   0];
d0 = [0, 27, 59, 91, 131, 141, 163, 185, 195, 275, 307, 339, 357, 390, 440];
a = a0 / max(a0);
d = round(d0 * L / d0(15));
d(15) = L;
for i = 1:14
    m = d(i) : d(i+1) - 1;
    slope = (a(i+1) - a(i)) / (d(i+1) - d(i));
    x(m+1) = a(i) + slope * (m - d(i));
end
end
