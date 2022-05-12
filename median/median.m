addpath('C:\MyProjects\cuFilters\bin\Release');
libname = 'cuFilters';
loadlibrary(libname,'cuFilters.h');
dims = [512,512,512];
tic
img = rand(dims,'single');
toc
subplot(1,3,1);
imshow(img(:,:,256));
title('rand');
output_ptr = libpointer('singlePtr',zeros(dims));

disp '3x3x3';
tic;
calllib(libname,'medianFilter3D',img,output_ptr,dims,3,'mirror')
fprintf('cuda:%f\n',toc);
cuda = reshape(output_ptr.value,dims);
tic
matlab = medfilt3(img,'symmetric');
fprintf('matlab:%f\n',toc);
diff  = cuda - matlab;
fprintf('diff:%f\n',sum(diff(:)));
subplot(1,3,2);
imshow(cuda(:,:,256));
title('3x3x3');

disp '5x5x5';
tic;
calllib(libname,'medianFilter3D',img,output_ptr,dims,5,'clamp')
fprintf('cuda:%f\n',toc);
cuda = reshape(output_ptr.value,dims);
tic
matlab = medfilt3(img,[5,5,5],'replicate');
fprintf('matlab:%f\n',toc);
diff  = cuda - matlab;
fprintf('diff:%f\n',sum(diff(:)));
subplot(1,3,3);
imshow(cuda(:,:,256));
title('5x5x5');


unloadlibrary(libname);