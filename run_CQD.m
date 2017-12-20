% function run_CQD()
% Final version for release
setup;
gpuidx = [1];

% Select dataset
% For example, you can do distillation:
% from 'cubcrop' to 'cub' for localization
% from 'cubcrop' to 'cublowres' for resolution
% from 'cubcrop' to 'cubedges' for edges
% from 'cubcrop' to 'cubdistort' for distortion
% from 'cubcrop' to 'cubcroplowres' for localization+resolution
% from 'cars' to 'carslowres' for resolution
% from 'cars' to 'carsedges' for edges
% Above are the experiments in Table 1 in the paper.
dirName = 'cub-localization';% Name of the directory to save the results
dataset_A = 'cubcrop';% Dataset distilling from
dataset_B = 'cub';% Dataset distilling to
if ~exist(['data/',dirName], 'dir')
    mkdir(['data/',dirName])
end
fileID = fopen(['data/',dirName,'/result.txt'],'a');

% Setting parameters
if strcmp(dataset_A(1:3),'cub')
    opts.lambda = 200;% Set lambda=200 for cub
    opts.learningRate = [0.0005*ones(1,5), 0.0005:-0.00005:0.0001 0.0001*ones(1,10), 0.00005*ones(1,10)]; % learning rate for CUB and distilling verydeep
elseif strcmp(dataset_A(1:4),'cars')
    opts.lambda = 50;% Set lambda=50 for cars
    opts.learningRate = [0.001*ones(1,5), (0.001:-0.0001:0.0001), 0.0001*ones(1, 10), 0.00005*ones(1,10)]; % learning rate for CARS
end
opts.T1 = 10;
opts.T2 = 10;
opts.useVal = 0; % if set to 0 then val is test
opts.numEpochs = 30;
opts.model = 'vgg-m'; % or vgg-vd
opts.batchSize = 128; % 32 for vgg-vd

%% 1. Train A test on A
opts.useDistill = 0;
opts.useVgg = 1;% use pre-trained model
run_train(gpuidx, dataset_A, [], dirName, 'trainA', opts);
imdb_A = load(['data/',dirName,'/trainA/',dataset_A,'-seed-01/imdb/imdb-seed-1.mat']);
net_A = load(['data/',dirName,'/trainA/',dataset_A,'-seed-01/net-deployed.mat']);
opts_A = load(['data/',dirName,'/trainA/',dataset_A,'-seed-01/opts.mat']);
acc = run_test(imdb_A, net_A, gpuidx, opts_A);
fprintf(fileID, 'train A test A : accuracy = %.2f%%\n', acc*100);
clear net_A;

%% 2. Train B test on B
opts.useDistill = 0;
opts.useVgg = 1;% use pre-trained model
run_train(gpuidx, dataset_B, [], dirName, 'trainB', opts);
imdb_B = load(['data/',dirName,'/trainB/',dataset_B,'-seed-01/imdb/imdb-seed-1.mat']);
net_B = load(['data/',dirName,'/trainB/',dataset_B,'-seed-01/net-deployed.mat']);
opts_B = load(['data/',dirName,'/trainB/',dataset_B,'-seed-01/opts.mat']);
acc = run_test(imdb_B, net_B, gpuidx, opts_B);
fprintf(fileID, 'train B test B : accuracy = %.2f%%\n', acc*100);
clear net_B;

%% 3. Train A test on B
net_A = load(['data/',dirName,'/trainA/',dataset_A,'-seed-01/net-deployed.mat']);
acc = run_test(imdb_B, net_A, gpuidx, opts_B);
fprintf(fileID, 'train A test B : accuracy = %.2f%%\n', acc*100);
clear net_A;

%% 4. Train A,B test on B
opts.useDistill = 0;
opts.useVgg = 0;% use model A
run_train(gpuidx, dataset_B, dataset_A, dirName, 'trainAtuneB', opts);
net_AB = load(['data/',dirName,'/trainAtuneB/',dataset_B,'-seed-01/net-deployed.mat']);
acc = run_test(imdb_B, net_AB, gpuidx, opts_B);
fprintf(fileID, 'train A,B test B : accuracy = %.2f%%\n', acc*100);
clear net_AB;

%% generate predictions from A
imdb_A = load(['data/',dirName,'/trainA/',dataset_A,'-seed-01/imdb/imdb-seed-1.mat']);
net_A = load(['data/',dirName,'/trainA/',dataset_A,'-seed-01/net-deployed.mat']);
opts_A = load(['data/',dirName,'/trainA/',dataset_A,'-seed-01/opts.mat']);
generate_prediction(imdb_A, opts_A, net_A, ['data/',dirName,'/',dataset_A,'/netOutput']);
clear net_A;

%% add netOutput to imdb
folder_name = 'CQD';
imdbDir_distill = ['data/',dirName,'/',folder_name,'/',dataset_B,'-seed-01/imdb'];
if ~exist([imdbDir_distill '/imdb-seed-1.mat'], 'file')
    imdb_B = load(['data/',dirName,'/trainB/',dataset_B,'-seed-01/imdb/imdb-seed-1.mat']);
    imdb_B.netOutputDir{1} = ['data/',dirName,'/',dataset_A,'/netOutput'];
    mkdir(imdbDir_distill);
    save([imdbDir_distill, '/imdb-seed-1.mat'], '-struct', 'imdb_B');
else
    imdb_B = load([imdbDir_distill, '/imdb-seed-1.mat']);
end
folder_name = 'A-CQD';
imdbDir_distill = ['data/',dirName,'/',folder_name,'/',dataset_B,'-seed-01/imdb'];
mkdir(imdbDir_distill);
save([imdbDir_distill, '/imdb-seed-1.mat'], '-struct', 'imdb_B');

%% 5. Train CQD test on B % not sure if it's working now
opts.useDistill = 1;
opts.useVgg = 1;% use pre-trained model
run_train(gpuidx, dataset_B, [], dirName, 'CQD', opts);
net_CQD = load(['data/',dirName,'/CQD/',dataset_B,'-seed-01/net-deployed.mat']);
acc = run_test(imdb_B, net_CQD, gpuidx, opts_B);
fprintf(fileID, 'train CQD test B : accuracy = %.2f%%\n', acc*100);
clear net_CQD;

%% 6. Train A,CQD test on B % not sure if it's working now
opts.useDistill = 1;
opts.useVgg = 0;% use model A
run_train(gpuidx, dataset_B, dataset_A, dirName, 'A-CQD', opts);
net_ACQD = load(['data/',dirName,'/A-CQD/',dataset_B,'-seed-01/net-deployed.mat']);
acc = run_test(imdb_B, net_ACQD, gpuidx, opts_B);
fprintf(fileID, 'train A,CQD test B : accuracy = %.2f%%\n', acc*100);
clear net_ACQD;

%% create dataset A+B
imdb_A = load(['data/',dirName,'/trainA/',dataset_A,'-seed-01/imdb/imdb-seed-1.mat']);
imdb_B = load(['data/',dirName,'/trainB/',dataset_B,'-seed-01/imdb/imdb-seed-1.mat']);
dataset_AB = [dataset_A,'_',dataset_B];
opts.imdbDir = ['data/',dirName,'/',dataset_AB,'-seed-01/imdb'];
if ~exist([opts.imdbDir '/imdb-seed-1.mat'], 'file')
    imdb_AB = combine_dataset(imdb_A, imdb_B);
    mkdir(opts.imdbDir);
    save([opts.imdbDir '/imdb-seed-1.mat'], '-struct', 'imdb_AB');
else
    imdb_AB = load([opts.imdbDir '/imdb-seed-1.mat']);
end

%% 7. Train A+B test on B
opts.useDistill = 0;
opts.useVgg = 1;% use pre-trained model
run_train(gpuidx, dataset_AB, [], dirName, 'A+B', opts); 
net_A_plus_B = load(['data/',dirName,'/A+B/',dataset_AB,'-seed-01/net-deployed.mat']);
acc = run_test(imdb_B, net_A_plus_B, gpuidx, opts_B);
fprintf(fileID, 'train A+B test B : accuracy = %.2f%%\n', acc*100);
clear net_A_plus_B;

fclose(fileID);
