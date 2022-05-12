function result = test_median
initialize;
tests = functiontests(localfunctions);
result = run(tests);
finalize;
end

%pseudo local variables
function name = libname
name = 'cuFilters';
end
function img = img_64
persistent img_64_
if isempty(img_64_)
    img_64_ = rand([64,64,64],'single');
end
img = img_64_;
end
function img = img_65
persistent img_65_
if isempty(img_65_)
    img_65_ = rand([64,64,65],'single');
end
img = img_65_;
end
function img = img_128
persistent img_128_
if isempty(img_128_)
    img_128_ = rand([512,512,128],'single');
end
img = img_128_;
end
function img = img_512
persistent img_512_
if isempty(img_512_)
    img_512_ = rand([512,512,512],'single');
end
img = img_512_;
end
function img = img_513
persistent img_513_
if isempty(img_513_)
    img_513_ = rand([512,512,513],'single');
end
img = img_513_;
end

%setup and teardown
function initialize()
disp 'setup'
loadlibrary(libname,'cuFilters.h');
end
function finalize()
disp 'teardown'
unloadlibrary(libname);
end

%functions for tests
function verify_medfilter(testCase, img, filter_size, texture_mode, padopt)
dims = size(img);
output_ptr = libpointer('singlePtr',zeros(dims));
tic
calllib(libname,'medianFilter3D',img,output_ptr,dims,filter_size,texture_mode)
cuda = reshape(output_ptr.value,dims);
fprintf('cuda : %f\n',toc);
tic
matlab = medfilt3(img,[1,1,1]*filter_size, padopt);
fprintf('matlab : %f\n',toc);
verifyEqual(testCase, cuda, matlab)
% fprintf('end of verification\n')
end

function test_3x3x3_z64(testCase)
disp '3x3x3 z64';
verify_medfilter(testCase, img_64, 3, 'mirror', 'symmetric');
end
function test_3x3x3_z65(testCase)
disp '3x3x3 z65';
verify_medfilter(testCase, img_65, 3, 'mirror', 'symmetric');
end

function test_5x5x5_z64(testCase)
disp '5x5x5 z64';
verify_medfilter(testCase, img_64, 5, 'clamp', 'replicate');
end
function test_5x5x5_z65(testCase)
disp '5x5x5 z65';
verify_medfilter(testCase, img_65, 5, 'clamp', 'replicate');
end

function test_7x7x7_z64(testCase)
disp '7x7x7 z64';
verify_medfilter(testCase, img_64, 7, 'clamp', 'replicate');
end
function test_7x7x7_z65(testCase)
disp '7x7x7 z65';
verify_medfilter(testCase, img_65, 7, 'clamp', 'replicate');
end
