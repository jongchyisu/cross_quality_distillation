function imdb3 = combine_dataset(imdb1, imdb2)
%% Concat two dataset imdb
    imdb3 = imdb1;
    
    %% images
    imdb3.images.name = [imdb1.images.name; imdb2.images.name];
    imdb3.images.id = [imdb1.images.id, imdb1.images.id(end) + imdb2.images.id]; % accumulate id
    imdb3.images.label = [imdb1.images.label, imdb2.images.label]; % class label
    imdb3.images.set = [imdb1.images.set, imdb2.images.set];
    imdb3.images.difficult = [imdb1.images.difficult, imdb2.images.difficult];

    %% Add this for getting images from correct directory 
    if ~isfield(imdb3.images,'number') % if only one dataset
        imdb3.images.ds_num = 1;
        imdb3.images.ds = ones(1, size(imdb1.images.id,2));   
    end
    imdb3.images.ds_num = imdb3.images.ds_num + 1;
    imdb3.images.ds = [imdb3.images.ds, imdb3.images.ds_num * ones(1, size(imdb2.images.id,2))];
    
    %% class
    imdb3.classes.name = cat(1,imdb1.classes.name,imdb2.classes.name);
    
    %% meta
    imdb3.meta.classes = cat(1,imdb1.meta.classes,imdb2.meta.classes);
    imdb3.meta.inUse = cat(2,imdb1.meta.inUse,imdb2.meta.inUse);
    
    %% imageDir
    if ischar(imdb3.imageDir)
        imdb3.imageDir = {imdb3.imageDir};
    end
    imdb3.imageDir{end+1} = imdb2.imageDir;

end