??
?=?=
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
?
AsString

input"T

output"
Ttype:
2		
"
	precisionint?????????"

scientificbool( "
shortestbool( "
widthint?????????"
fillstring 
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
S
	Bucketize

input"T

output"
Ttype:
2	"

boundarieslist(float)
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
b
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
SparseCross
indices	*N
values2sparse_types
shapes	*N
dense_inputs2dense_types
output_indices	
output_values"out_type
output_shape	"

Nint("
hashed_outputbool"
num_bucketsint("
hash_keyint"$
sparse_types
list(type)(:
2	"#
dense_types
list(type)(:
2	"
out_typetype:
2	"
internal_typetype:
2	
?
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
?
SparseSegmentSqrtN	
data"T
indices"Tidx
segment_ids"Tsegmentids
output"T"
Ttype:
2"
Tidxtype0:
2	"
Tsegmentidstype0:
2	
?
SparseSegmentSum	
data"T
indices"Tidx
segment_ids"Tsegmentids
output"T"
Ttype:
2	"
Tidxtype0:
2	"
Tsegmentidstype0:
2	
?
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
9
VarIsInitializedOp
resource
is_initialized
?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.4.12v2.4.0-49-g85c8b2a817f8??

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
k
global_step
VariableV2*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: 
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
o
input_example_tensorPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
U
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
W
ParseExample/Const_4Const*
_output_shapes
: *
dtype0*
valueB 
d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB 
?
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
:*
dtype0*S
valueJBHB	educationBmarital_statusB
occupationBrelationshipB	workclass
?
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:*
dtype0*S
valueJBHBageBcapital_gainBcapital_lossBeducation_numBhours_per_week
j
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB 
?
ParseExample/ParseExampleV2ParseExampleV2input_example_tensor!ParseExample/ParseExampleV2/names'ParseExample/ParseExampleV2/sparse_keys&ParseExample/ParseExampleV2/dense_keys'ParseExample/ParseExampleV2/ragged_keysParseExample/ConstParseExample/Const_1ParseExample/Const_2ParseExample/Const_3ParseExample/Const_4*
Tdense	
2*?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::?????????:?????????:?????????:?????????:?????????*0
dense_shapes 
:::::*

num_sparse*
ragged_split_types
 *
ragged_value_types
 *
sparse_types	
2
?
ydnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/shapeConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes
:*
dtype0*
valueB"      
?
wdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/minConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes
: *
dtype0*
valueB
 *?5?
?
wdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/maxConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes
: *
dtype0*
valueB
 *?5?
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/RandomUniformRandomUniformydnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/shape*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:*
dtype0
?
wdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/subSubwdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/maxwdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/min*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes
: 
?
wdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/mulMul?dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/RandomUniformwdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/sub*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:
?
sdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniformAddwdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/mulwdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform/min*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:
?
Xdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0
VariableV2*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:*
dtype0*
shape
:
?
_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/AssignAssignXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0sdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:
?
]dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/readIdentityXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:
?
odnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/Initializer/zerosConst*p
_classf
dbloc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0*
_output_shapes
:*
dtype0	*
valueB	R 
?
]dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0
VariableV2*p
_classf
dbloc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0*
_output_shapes
:*
dtype0	*
shape:
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/AssignAssign]dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0odnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/Initializer/zeros*
T0	*p
_classf
dbloc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0*
_output_shapes
:
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/readIdentity]dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0*
T0	*p
_classf
dbloc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0*
_output_shapes
:
?
idnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/Initializer/zerosConst*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
?
Wdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0
VariableV2*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0*
_output_shapes

:*
dtype0*
shape
:
?
^dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/AssignAssignWdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0idnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/Initializer/zeros*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0*
_output_shapes

:
?
\dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/readIdentityWdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0*
_output_shapes

:
?
ndnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/Initializer/zerosConst*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0*
_output_shapes
:*
dtype0	*
valueB	R 
?
\dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0
VariableV2*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0*
_output_shapes
:*
dtype0	*
shape:
?
cdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/AssignAssign\dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0ndnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/Initializer/zeros*
T0	*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0*
_output_shapes
:
?
adnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/readIdentity\dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0*
T0	*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0*
_output_shapes
:
?
4dnn/input_from_feature_columns/input_layer/age/ShapeShapeParseExample/ParseExampleV2:15*
T0*
_output_shapes
:
?
Bdnn/input_from_feature_columns/input_layer/age/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ddnn/input_from_feature_columns/input_layer/age/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ddnn/input_from_feature_columns/input_layer/age/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
<dnn/input_from_feature_columns/input_layer/age/strided_sliceStridedSlice4dnn/input_from_feature_columns/input_layer/age/ShapeBdnn/input_from_feature_columns/input_layer/age/strided_slice/stackDdnn/input_from_feature_columns/input_layer/age/strided_slice/stack_1Ddnn/input_from_feature_columns/input_layer/age/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
>dnn/input_from_feature_columns/input_layer/age/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
<dnn/input_from_feature_columns/input_layer/age/Reshape/shapePack<dnn/input_from_feature_columns/input_layer/age/strided_slice>dnn/input_from_feature_columns/input_layer/age/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
6dnn/input_from_feature_columns/input_layer/age/ReshapeReshapeParseExample/ParseExampleV2:15<dnn/input_from_feature_columns/input_layer/age/Reshape/shape*
T0*'
_output_shapes
:?????????
?
=dnn/input_from_feature_columns/input_layer/capital_gain/ShapeShapeParseExample/ParseExampleV2:16*
T0*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Mdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Mdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Ednn/input_from_feature_columns/input_layer/capital_gain/strided_sliceStridedSlice=dnn/input_from_feature_columns/input_layer/capital_gain/ShapeKdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stackMdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stack_1Mdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Gdnn/input_from_feature_columns/input_layer/capital_gain/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Ednn/input_from_feature_columns/input_layer/capital_gain/Reshape/shapePackEdnn/input_from_feature_columns/input_layer/capital_gain/strided_sliceGdnn/input_from_feature_columns/input_layer/capital_gain/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/capital_gain/ReshapeReshapeParseExample/ParseExampleV2:16Ednn/input_from_feature_columns/input_layer/capital_gain/Reshape/shape*
T0*'
_output_shapes
:?????????
?
=dnn/input_from_feature_columns/input_layer/capital_loss/ShapeShapeParseExample/ParseExampleV2:17*
T0*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Mdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Mdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Ednn/input_from_feature_columns/input_layer/capital_loss/strided_sliceStridedSlice=dnn/input_from_feature_columns/input_layer/capital_loss/ShapeKdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stackMdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stack_1Mdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Gdnn/input_from_feature_columns/input_layer/capital_loss/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Ednn/input_from_feature_columns/input_layer/capital_loss/Reshape/shapePackEdnn/input_from_feature_columns/input_layer/capital_loss/strided_sliceGdnn/input_from_feature_columns/input_layer/capital_loss/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/capital_loss/ReshapeReshapeParseExample/ParseExampleV2:17Ednn/input_from_feature_columns/input_layer/capital_loss/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Udnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/ConstConst*
_output_shapes
:*
dtype0*?
value?B?B	BachelorsBHS-gradB11thBMastersB9thBSome-collegeB
Assoc-acdmB	Assoc-vocB7th-8thB	DoctorateBProf-schoolB5th-6thB10thB1st-4thB	PreschoolB12th
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
[dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
[dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
Udnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/rangeRange[dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range/startTdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/Size[dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range/delta*
_output_shapes
:
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/CastCastUdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
`dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
ednn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_bd125ad8-cd73-4281-8aad-e99c7897566e*
value_dtype0	
?
ydnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2ednn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/hash_tableUdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/ConstTdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/Cast*	
Tin0*

Tout0	
?
bdnn/input_from_feature_columns/input_layer/education_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2ednn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/hash_tableParseExample/ParseExampleV2:5`dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
Zdnn/input_from_feature_columns/input_layer/education_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/SparseToDenseSparseToDenseParseExample/ParseExampleV2ParseExample/ParseExampleV2:10bdnn/input_from_feature_columns/input_layer/education_indicator/hash_table_Lookup/LookupTableFindV2Zdnn/input_from_feature_columns/input_layer/education_indicator/SparseToDense/default_value*
T0	*
Tindices0	*0
_output_shapes
:??????????????????
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Ndnn/input_from_feature_columns/input_layer/education_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
?
Fdnn/input_from_feature_columns/input_layer/education_indicator/one_hotOneHotLdnn/input_from_feature_columns/input_layer/education_indicator/SparseToDenseLdnn/input_from_feature_columns/input_layer/education_indicator/one_hot/depthLdnn/input_from_feature_columns/input_layer/education_indicator/one_hot/ConstNdnn/input_from_feature_columns/input_layer/education_indicator/one_hot/Const_1*
T0*4
_output_shapes"
 :??????????????????
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Bdnn/input_from_feature_columns/input_layer/education_indicator/SumSumFdnn/input_from_feature_columns/input_layer/education_indicator/one_hotTdnn/input_from_feature_columns/input_layer/education_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
?
Ddnn/input_from_feature_columns/input_layer/education_indicator/ShapeShapeBdnn/input_from_feature_columns/input_layer/education_indicator/Sum*
T0*
_output_shapes
:
?
Rdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/education_indicator/ShapeRdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stackTdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Ndnn/input_from_feature_columns/input_layer/education_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/education_indicator/strided_sliceNdnn/input_from_feature_columns/input_layer/education_indicator/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Fdnn/input_from_feature_columns/input_layer/education_indicator/ReshapeReshapeBdnn/input_from_feature_columns/input_layer/education_indicator/SumLdnn/input_from_feature_columns/input_layer/education_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????
?
>dnn/input_from_feature_columns/input_layer/education_num/ShapeShapeParseExample/ParseExampleV2:18*
T0*
_output_shapes
:
?
Ldnn/input_from_feature_columns/input_layer/education_num/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ndnn/input_from_feature_columns/input_layer/education_num/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ndnn/input_from_feature_columns/input_layer/education_num/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Fdnn/input_from_feature_columns/input_layer/education_num/strided_sliceStridedSlice>dnn/input_from_feature_columns/input_layer/education_num/ShapeLdnn/input_from_feature_columns/input_layer/education_num/strided_slice/stackNdnn/input_from_feature_columns/input_layer/education_num/strided_slice/stack_1Ndnn/input_from_feature_columns/input_layer/education_num/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Hdnn/input_from_feature_columns/input_layer/education_num/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Fdnn/input_from_feature_columns/input_layer/education_num/Reshape/shapePackFdnn/input_from_feature_columns/input_layer/education_num/strided_sliceHdnn/input_from_feature_columns/input_layer/education_num/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
@dnn/input_from_feature_columns/input_layer/education_num/ReshapeReshapeParseExample/ParseExampleV2:18Fdnn/input_from_feature_columns/input_layer/education_num/Reshape/shape*
T0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/hours_per_week/ShapeShapeParseExample/ParseExampleV2:19*
T0*
_output_shapes
:
?
Mdnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Odnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Odnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Gdnn/input_from_feature_columns/input_layer/hours_per_week/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/hours_per_week/ShapeMdnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stackOdnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Idnn/input_from_feature_columns/input_layer/hours_per_week/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Gdnn/input_from_feature_columns/input_layer/hours_per_week/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/hours_per_week/strided_sliceIdnn/input_from_feature_columns/input_layer/hours_per_week/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Adnn/input_from_feature_columns/input_layer/hours_per_week/ReshapeReshapeParseExample/ParseExampleV2:19Gdnn/input_from_feature_columns/input_layer/hours_per_week/Reshape/shape*
T0*'
_output_shapes
:?????????
?
_dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/ConstConst*
_output_shapes
:*
dtype0*~
valueuBsBMarried-civ-spouseBDivorcedBMarried-spouse-absentBNever-marriedB	SeparatedBMarried-AF-spouseBWidowed
?
^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
ednn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
ednn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
_dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/rangeRangeednn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range/start^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/Sizeednn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range/delta*
_output_shapes
:
?
^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/CastCast_dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
odnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_1671b855-9658-400c-ba5e-749bdb1f53c1*
value_dtype0	
?
?dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2odnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/hash_table_dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/Const^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/Cast*	
Tin0*

Tout0	
?
gdnn/input_from_feature_columns/input_layer/marital_status_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2odnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/hash_tableParseExample/ParseExampleV2:6jdnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
_dnn/input_from_feature_columns/input_layer/marital_status_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/SparseToDenseSparseToDenseParseExample/ParseExampleV2:1ParseExample/ParseExampleV2:11gdnn/input_from_feature_columns/input_layer/marital_status_indicator/hash_table_Lookup/LookupTableFindV2_dnn/input_from_feature_columns/input_layer/marital_status_indicator/SparseToDense/default_value*
T0	*
Tindices0	*0
_output_shapes
:??????????????????
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Sdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
?
Kdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hotOneHotQdnn/input_from_feature_columns/input_layer/marital_status_indicator/SparseToDenseQdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/depthQdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/ConstSdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/Const_1*
T0*4
_output_shapes"
 :??????????????????
?
Ydnn/input_from_feature_columns/input_layer/marital_status_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Gdnn/input_from_feature_columns/input_layer/marital_status_indicator/SumSumKdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hotYdnn/input_from_feature_columns/input_layer/marital_status_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
?
Idnn/input_from_feature_columns/input_layer/marital_status_indicator/ShapeShapeGdnn/input_from_feature_columns/input_layer/marital_status_indicator/Sum*
T0*
_output_shapes
:
?
Wdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ydnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ydnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_sliceStridedSliceIdnn/input_from_feature_columns/input_layer/marital_status_indicator/ShapeWdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stackYdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stack_1Ydnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Sdnn/input_from_feature_columns/input_layer/marital_status_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/Reshape/shapePackQdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_sliceSdnn/input_from_feature_columns/input_layer/marital_status_indicator/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/marital_status_indicator/ReshapeReshapeGdnn/input_from_feature_columns/input_layer/marital_status_indicator/SumQdnn/input_from_feature_columns/input_layer/marital_status_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Fdnn/input_from_feature_columns/input_layer/occupation_embedding/lookupStringToHashBucketFastParseExample/ParseExampleV2:7*#
_output_shapes
:?????????*
num_buckets
?
Vdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_mapsIdentitybdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/read*
T0	*
_output_shapes
:
?
Mdnn/input_from_feature_columns/input_layer/occupation_embedding/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Hdnn/input_from_feature_columns/input_layer/occupation_embedding/GatherV2GatherV2Vdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_mapsFdnn/input_from_feature_columns/input_layer/occupation_embedding/lookupMdnn/input_from_feature_columns/input_layer/occupation_embedding/GatherV2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
hdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SliceSliceParseExample/ParseExampleV2:12hdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice/begingdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
adnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/ProdProdbdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slicebdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Const*
T0	*
_output_shapes
: 
?
mdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
ednn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2GatherV2ParseExample/ParseExampleV2:12mdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/indicesjdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
cdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Cast/xPackadnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Prodednn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleV2:2ParseExample/ParseExampleV2:12cdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
sdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape/IdentityIdentityHdnn/input_from_feature_columns/input_layer/occupation_embedding/GatherV2*
T0	*#
_output_shapes
:?????????
?
kdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
idnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqualGreaterEqualsdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape/Identitykdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/WhereWhereidnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/ReshapeReshapebdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Wherejdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
ldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1GatherV2jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshapeddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
ldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2GatherV2sdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape/Identityddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
ednn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/IdentityIdentityldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
?
vdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsgdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2ednn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Identityvdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
{dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes
: *
dtype0*
value	B : 
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2]dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/read{dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/Unique?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:?????????
?
tdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparseSparseSegmentSqrtN?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity}dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/Unique:1?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
ldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
fdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/ShapeShapetdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
pdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
rdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
rdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Shapepdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stackrdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_1rdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/stackPackddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/stack/0jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
?
adnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/TileTilefdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/zeros_like	ZerosLiketdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
\dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weightsSelectadnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Tilegdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/zeros_liketdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
cdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Cast_1CastParseExample/ParseExampleV2:12*

DstT0*

SrcT0	*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
idnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1Slicecdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Cast_1jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/beginidnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Shape_1Shape\dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights*
T0*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
idnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2Sliceddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Shape_1jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/beginidnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
cdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/concatConcatV2ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2hdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
?
fdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_2Reshape\dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weightscdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/concat*
T0*'
_output_shapes
:?????????
?
Ednn/input_from_feature_columns/input_layer/occupation_embedding/ShapeShapefdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_2*
T0*
_output_shapes
:
?
Sdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Udnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Udnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Mdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/occupation_embedding/ShapeSdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stackUdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Odnn/input_from_feature_columns/input_layer/occupation_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Mdnn/input_from_feature_columns/input_layer/occupation_embedding/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_sliceOdnn/input_from_feature_columns/input_layer/occupation_embedding/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Gdnn/input_from_feature_columns/input_layer/occupation_embedding/ReshapeReshapefdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_2Mdnn/input_from_feature_columns/input_layer/occupation_embedding/Reshape/shape*
T0*'
_output_shapes
:?????????
?
[dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/ConstConst*
_output_shapes
:*
dtype0*W
valueNBLBHusbandBNot-in-familyBWifeB	Own-childB	UnmarriedBOther-relative
?
Zdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
adnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
adnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
[dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/rangeRangeadnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range/startZdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/Sizeadnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range/delta*
_output_shapes
:
?
Zdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/CastCast[dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
fdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
kdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_4c87fd7a-c02a-4a86-8bf4-58228587a0cc*
value_dtype0	
?
dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2kdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/hash_table[dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/ConstZdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/Cast*	
Tin0*

Tout0	
?
ednn/input_from_feature_columns/input_layer/relationship_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2kdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/hash_tableParseExample/ParseExampleV2:8fdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
]dnn/input_from_feature_columns/input_layer/relationship_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/SparseToDenseSparseToDenseParseExample/ParseExampleV2:3ParseExample/ParseExampleV2:13ednn/input_from_feature_columns/input_layer/relationship_indicator/hash_table_Lookup/LookupTableFindV2]dnn/input_from_feature_columns/input_layer/relationship_indicator/SparseToDense/default_value*
T0	*
Tindices0	*0
_output_shapes
:??????????????????
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Qdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
?
Idnn/input_from_feature_columns/input_layer/relationship_indicator/one_hotOneHotOdnn/input_from_feature_columns/input_layer/relationship_indicator/SparseToDenseOdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/depthOdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/ConstQdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/Const_1*
T0*4
_output_shapes"
 :??????????????????
?
Wdnn/input_from_feature_columns/input_layer/relationship_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Ednn/input_from_feature_columns/input_layer/relationship_indicator/SumSumIdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hotWdnn/input_from_feature_columns/input_layer/relationship_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
?
Gdnn/input_from_feature_columns/input_layer/relationship_indicator/ShapeShapeEdnn/input_from_feature_columns/input_layer/relationship_indicator/Sum*
T0*
_output_shapes
:
?
Udnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Wdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Wdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/strided_sliceStridedSliceGdnn/input_from_feature_columns/input_layer/relationship_indicator/ShapeUdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stackWdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stack_1Wdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Qdnn/input_from_feature_columns/input_layer/relationship_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/Reshape/shapePackOdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_sliceQdnn/input_from_feature_columns/input_layer/relationship_indicator/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Idnn/input_from_feature_columns/input_layer/relationship_indicator/ReshapeReshapeEdnn/input_from_feature_columns/input_layer/relationship_indicator/SumOdnn/input_from_feature_columns/input_layer/relationship_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Ednn/input_from_feature_columns/input_layer/workclass_embedding/lookupStringToHashBucketFastParseExample/ParseExampleV2:9*#
_output_shapes
:?????????*
num_buckets
?
Udnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_mapsIdentityadnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/read*
T0	*
_output_shapes
:
?
Ldnn/input_from_feature_columns/input_layer/workclass_embedding/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Gdnn/input_from_feature_columns/input_layer/workclass_embedding/GatherV2GatherV2Udnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_mapsEdnn/input_from_feature_columns/input_layer/workclass_embedding/lookupLdnn/input_from_feature_columns/input_layer/workclass_embedding/GatherV2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
fdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
ednn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SliceSliceParseExample/ParseExampleV2:14fdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice/beginednn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
_dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/ProdProd`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Const*
T0	*
_output_shapes
: 
?
kdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
cdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2GatherV2ParseExample/ParseExampleV2:14kdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2/indiceshdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
adnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Cast/xPack_dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Prodcdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2*
N*
T0	*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseReshapeSparseReshapeParseExample/ParseExampleV2:4ParseExample/ParseExampleV2:14adnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
qdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseReshape/IdentityIdentityGdnn/input_from_feature_columns/input_layer/workclass_embedding/GatherV2*
T0	*#
_output_shapes
:?????????
?
idnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
gdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GreaterEqualGreaterEqualqdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseReshape/Identityidnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/WhereWheregdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
bdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/ReshapeReshape`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Wherehdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
ednn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2_1GatherV2hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseReshapebdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
ednn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2_2GatherV2qdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseReshape/Identitybdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshapejdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
cdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/IdentityIdentityjdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
?
tdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsednn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2_1ednn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/GatherV2_2cdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Identitytdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
ydnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0*
_output_shapes
: *
dtype0*
value	B : 
?
?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2\dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/readydnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/Unique?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis*
Taxis0*
Tindices0	*
Tparams0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*'
_output_shapes
:?????????
?
rdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparseSparseSegmentSqrtN?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity{dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/Unique:1?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
ddnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2jdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/ShapeShaperdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
ndnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
pdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
pdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/strided_sliceStridedSlice`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Shapendnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/strided_slice/stackpdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/strided_slice/stack_1pdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
bdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/stackPackbdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/stack/0hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/strided_slice*
N*
T0*
_output_shapes
:
?
_dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/TileTileddnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshape_1`dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
ednn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/zeros_like	ZerosLikerdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Zdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weightsSelect_dnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Tileednn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/zeros_likerdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
adnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Cast_1CastParseExample/ParseExampleV2:14*

DstT0*

SrcT0	*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
gdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
bdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_1Sliceadnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Cast_1hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_1/begingdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
bdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Shape_1ShapeZdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights*
T0*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
gdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
bdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_2Slicebdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Shape_1hdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_2/begingdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
fdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
adnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/concatConcatV2bdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_1bdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Slice_2fdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/concat/axis*
N*
T0*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshape_2ReshapeZdnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weightsadnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/concat*
T0*'
_output_shapes
:?????????
?
Ddnn/input_from_feature_columns/input_layer/workclass_embedding/ShapeShapeddnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshape_2*
T0*
_output_shapes
:
?
Rdnn/input_from_feature_columns/input_layer/workclass_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Tdnn/input_from_feature_columns/input_layer/workclass_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Tdnn/input_from_feature_columns/input_layer/workclass_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Ldnn/input_from_feature_columns/input_layer/workclass_embedding/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/workclass_embedding/ShapeRdnn/input_from_feature_columns/input_layer/workclass_embedding/strided_slice/stackTdnn/input_from_feature_columns/input_layer/workclass_embedding/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/workclass_embedding/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Ndnn/input_from_feature_columns/input_layer/workclass_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Ldnn/input_from_feature_columns/input_layer/workclass_embedding/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/workclass_embedding/strided_sliceNdnn/input_from_feature_columns/input_layer/workclass_embedding/Reshape/shape/1*
N*
T0*
_output_shapes
:
?
Fdnn/input_from_feature_columns/input_layer/workclass_embedding/ReshapeReshapeddnn/input_from_feature_columns/input_layer/workclass_embedding/workclass_embedding_weights/Reshape_2Ldnn/input_from_feature_columns/input_layer/workclass_embedding/Reshape/shape*
T0*'
_output_shapes
:?????????
?
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
1dnn/input_from_feature_columns/input_layer/concatConcatV26dnn/input_from_feature_columns/input_layer/age/Reshape?dnn/input_from_feature_columns/input_layer/capital_gain/Reshape?dnn/input_from_feature_columns/input_layer/capital_loss/ReshapeFdnn/input_from_feature_columns/input_layer/education_indicator/Reshape@dnn/input_from_feature_columns/input_layer/education_num/ReshapeAdnn/input_from_feature_columns/input_layer/hours_per_week/ReshapeKdnn/input_from_feature_columns/input_layer/marital_status_indicator/ReshapeGdnn/input_from_feature_columns/input_layer/occupation_embedding/ReshapeIdnn/input_from_feature_columns/input_layer/relationship_indicator/ReshapeFdnn/input_from_feature_columns/input_layer/workclass_embedding/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
N
*
T0*'
_output_shapes
:?????????2
?
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:*
dtype0*
valueB"2   ?   
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *? <?
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *? <>
?
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	2?*
dtype0
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	2?
?
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	2?
?
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
dtype0*
shape:	2?*0
shared_name!dnn/hiddenlayer_0/kernel/part_0
?
@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
dtype0
?
3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	2?*
dtype0
?
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:?*
dtype0*
valueB?*    
?
dnn/hiddenlayer_0/bias/part_0VarHandleOp*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
: *
dtype0*
shape:?*.
shared_namednn/hiddenlayer_0/bias/part_0
?
>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
dtype0
?
1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:?*
dtype0
?
'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:	2?*
dtype0
w
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
T0*
_output_shapes
:	2?
?
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*(
_output_shapes
:??????????
?
%dnn/hiddenlayer_0/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes	
:?*
dtype0
o
dnn/hiddenlayer_0/biasIdentity%dnn/hiddenlayer_0/bias/ReadVariableOp*
T0*
_output_shapes	
:?
?
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*(
_output_shapes
:??????????
l
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*(
_output_shapes
:??????????
g
dnn/zero_fraction/SizeSizednn/hiddenlayer_0/Relu*
T0*
_output_shapes
: *
out_type0	
c
dnn/zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction/cond/SwitchSwitchdnn/zero_fraction/LessEqualdnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
m
dnn/zero_fraction/cond/switch_tIdentitydnn/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
k
dnn/zero_fraction/cond/switch_fIdentitydnn/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
h
dnn/zero_fraction/cond/pred_idIdentitydnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
*dnn/zero_fraction/cond/count_nonzero/zerosConst ^dnn/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
-dnn/zero_fraction/cond/count_nonzero/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1*dnn/zero_fraction/cond/count_nonzero/zeros*
T0*(
_output_shapes
:??????????
?
4dnn/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu*<
_output_shapes*
(:??????????:??????????
?
)dnn/zero_fraction/cond/count_nonzero/CastCast-dnn/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*(
_output_shapes
:??????????
?
*dnn/zero_fraction/cond/count_nonzero/ConstConst ^dnn/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
2dnn/zero_fraction/cond/count_nonzero/nonzero_countSum)dnn/zero_fraction/cond/count_nonzero/Cast*dnn/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
dnn/zero_fraction/cond/CastCast2dnn/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
,dnn/zero_fraction/cond/count_nonzero_1/zerosConst ^dnn/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
/dnn/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch,dnn/zero_fraction/cond/count_nonzero_1/zeros*
T0*(
_output_shapes
:??????????
?
6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu*<
_output_shapes*
(:??????????:??????????
?
+dnn/zero_fraction/cond/count_nonzero_1/CastCast/dnn/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*(
_output_shapes
:??????????
?
,dnn/zero_fraction/cond/count_nonzero_1/ConstConst ^dnn/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countSum+dnn/zero_fraction/cond/count_nonzero_1/Cast,dnn/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
dnn/zero_fraction/cond/MergeMerge4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countdnn/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
{
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values
?
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
T0*
_output_shapes
: 
?
$dnn/dnn/hiddenlayer_0/activation/tagConst*
_output_shapes
: *
dtype0*1
value(B& B dnn/dnn/hiddenlayer_0/activation
?
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
?
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:*
dtype0*
valueB"?   @   
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *?5?
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *?5>
?
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	?@*
dtype0
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	?@
?
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	?@
?
dnn/hiddenlayer_1/kernel/part_0VarHandleOp*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: *
dtype0*
shape:	?@*0
shared_name!dnn/hiddenlayer_1/kernel/part_0
?
@dnn/hiddenlayer_1/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_1/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*
dtype0
?
3dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	?@*
dtype0
?
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
:@*
dtype0*
valueB@*    
?
dnn/hiddenlayer_1/bias/part_0VarHandleOp*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
_output_shapes
: *
dtype0*
shape:@*.
shared_namednn/hiddenlayer_1/bias/part_0
?
>dnn/hiddenlayer_1/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_1/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*
dtype0
?
1dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
:@*
dtype0
?
'dnn/hiddenlayer_1/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
:	?@*
dtype0
w
dnn/hiddenlayer_1/kernelIdentity'dnn/hiddenlayer_1/kernel/ReadVariableOp*
T0*
_output_shapes
:	?@
?
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:?????????@

%dnn/hiddenlayer_1/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
:@*
dtype0
n
dnn/hiddenlayer_1/biasIdentity%dnn/hiddenlayer_1/bias/ReadVariableOp*
T0*
_output_shapes
:@
?
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*'
_output_shapes
:?????????@
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:?????????@
i
dnn/zero_fraction_1/SizeSizednn/hiddenlayer_1/Relu*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_1/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_1/cond/SwitchSwitchdnn/zero_fraction_1/LessEqualdnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_1/cond/switch_tIdentity!dnn/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_1/cond/switch_fIdentitydnn/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_1/cond/pred_idIdentitydnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: 
?
,dnn/zero_fraction_1/cond/count_nonzero/zerosConst"^dnn/zero_fraction_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
/dnn/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_1/cond/count_nonzero/zeros*
T0*'
_output_shapes
:?????????@
?
6dnn/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu*:
_output_shapes(
&:?????????@:?????????@
?
+dnn/zero_fraction_1/cond/count_nonzero/CastCast/dnn/zero_fraction_1/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*'
_output_shapes
:?????????@
?
,dnn/zero_fraction_1/cond/count_nonzero/ConstConst"^dnn/zero_fraction_1/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
4dnn/zero_fraction_1/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_1/cond/count_nonzero/Cast,dnn/zero_fraction_1/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
dnn/zero_fraction_1/cond/CastCast4dnn/zero_fraction_1/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
.dnn/zero_fraction_1/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_1/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_1/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:?????????@
?
8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu*:
_output_shapes(
&:?????????@:?????????@
?
-dnn/zero_fraction_1/cond/count_nonzero_1/CastCast1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*'
_output_shapes
:?????????@
?
.dnn/zero_fraction_1/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_1/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_1/cond/count_nonzero_1/Cast.dnn/zero_fraction_1/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_1/cond/MergeMerge6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_1/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Sizednn/zero_fraction_1/cond/Merge*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values
?
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
T0*
_output_shapes
: 
?
$dnn/dnn/hiddenlayer_1/activation/tagConst*
_output_shapes
: *
dtype0*1
value(B& B dnn/dnn/hiddenlayer_1/activation
?
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: 
?
@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
:*
dtype0*
valueB"@      
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *>???
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *>??>
?
Hdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:@*
dtype0
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:@
?
:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:@
?
dnn/hiddenlayer_2/kernel/part_0VarHandleOp*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!dnn/hiddenlayer_2/kernel/part_0
?
@dnn/hiddenlayer_2/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_2/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_2/kernel/part_0:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform*
dtype0
?
3dnn/hiddenlayer_2/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:@*
dtype0
?
/dnn/hiddenlayer_2/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
:*
dtype0*
valueB*    
?
dnn/hiddenlayer_2/bias/part_0VarHandleOp*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
_output_shapes
: *
dtype0*
shape:*.
shared_namednn/hiddenlayer_2/bias/part_0
?
>dnn/hiddenlayer_2/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_2/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_2/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_2/bias/part_0/dnn/hiddenlayer_2/bias/part_0/Initializer/zeros*
dtype0
?
1dnn/hiddenlayer_2/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0*
_output_shapes
:*
dtype0
?
'dnn/hiddenlayer_2/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:@*
dtype0
v
dnn/hiddenlayer_2/kernelIdentity'dnn/hiddenlayer_2/kernel/ReadVariableOp*
T0*
_output_shapes

:@
?
dnn/hiddenlayer_2/MatMulMatMuldnn/hiddenlayer_1/Reludnn/hiddenlayer_2/kernel*
T0*'
_output_shapes
:?????????

%dnn/hiddenlayer_2/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0*
_output_shapes
:*
dtype0
n
dnn/hiddenlayer_2/biasIdentity%dnn/hiddenlayer_2/bias/ReadVariableOp*
T0*
_output_shapes
:
?
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/bias*
T0*'
_output_shapes
:?????????
k
dnn/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*
T0*'
_output_shapes
:?????????
i
dnn/zero_fraction_2/SizeSizednn/hiddenlayer_2/Relu*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_2/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_2/cond/SwitchSwitchdnn/zero_fraction_2/LessEqualdnn/zero_fraction_2/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_2/cond/switch_tIdentity!dnn/zero_fraction_2/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_2/cond/switch_fIdentitydnn/zero_fraction_2/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_2/cond/pred_idIdentitydnn/zero_fraction_2/LessEqual*
T0
*
_output_shapes
: 
?
,dnn/zero_fraction_2/cond/count_nonzero/zerosConst"^dnn/zero_fraction_2/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
/dnn/zero_fraction_2/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_2/cond/count_nonzero/zeros*
T0*'
_output_shapes
:?????????
?
6dnn/zero_fraction_2/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_2/Relu dnn/zero_fraction_2/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_2/Relu*:
_output_shapes(
&:?????????:?????????
?
+dnn/zero_fraction_2/cond/count_nonzero/CastCast/dnn/zero_fraction_2/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*'
_output_shapes
:?????????
?
,dnn/zero_fraction_2/cond/count_nonzero/ConstConst"^dnn/zero_fraction_2/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
4dnn/zero_fraction_2/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_2/cond/count_nonzero/Cast,dnn/zero_fraction_2/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
dnn/zero_fraction_2/cond/CastCast4dnn/zero_fraction_2/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
.dnn/zero_fraction_2/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_2/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_2/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:?????????
?
8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_2/Relu dnn/zero_fraction_2/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_2/Relu*:
_output_shapes(
&:?????????:?????????
?
-dnn/zero_fraction_2/cond/count_nonzero_1/CastCast1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*'
_output_shapes
:?????????
?
.dnn/zero_fraction_2/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_2/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_2/cond/count_nonzero_1/Cast.dnn/zero_fraction_2/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_2/cond/MergeMerge6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_2/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Sizednn/zero_fraction_2/cond/Merge*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values
?
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
T0*
_output_shapes
: 
?
$dnn/dnn/hiddenlayer_2/activation/tagConst*
_output_shapes
: *
dtype0*1
value(B& B dnn/dnn/hiddenlayer_2/activation
?
 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tagdnn/hiddenlayer_2/Relu*
_output_shapes
: 
?
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
:*
dtype0*
valueB"      
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *????
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *???>
?
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:*
dtype0
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
?
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
?
dnn/logits/kernel/part_0VarHandleOp*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
dtype0*
shape
:*)
shared_namednn/logits/kernel/part_0
?
9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 
?
dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
dtype0
?
,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
_output_shapes

:*
dtype0
?
(dnn/logits/bias/part_0/Initializer/zerosConst*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:*
dtype0*
valueB*    
?
dnn/logits/bias/part_0VarHandleOp*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
: *
dtype0*
shape:*'
shared_namednn/logits/bias/part_0
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 
?
dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
dtype0
}
*dnn/logits/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
_output_shapes
:*
dtype0
y
 dnn/logits/kernel/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
_output_shapes

:*
dtype0
h
dnn/logits/kernelIdentity dnn/logits/kernel/ReadVariableOp*
T0*
_output_shapes

:
x
dnn/logits/MatMulMatMuldnn/hiddenlayer_2/Reludnn/logits/kernel*
T0*'
_output_shapes
:?????????
q
dnn/logits/bias/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
_output_shapes
:*
dtype0
`
dnn/logits/biasIdentitydnn/logits/bias/ReadVariableOp*
T0*
_output_shapes
:
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*'
_output_shapes
:?????????
e
dnn/zero_fraction_3/SizeSizednn/logits/BiasAdd*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_3/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
dnn/zero_fraction_3/LessEqual	LessEqualdnn/zero_fraction_3/Sizednn/zero_fraction_3/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_3/cond/SwitchSwitchdnn/zero_fraction_3/LessEqualdnn/zero_fraction_3/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_3/cond/switch_tIdentity!dnn/zero_fraction_3/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_3/cond/switch_fIdentitydnn/zero_fraction_3/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_3/cond/pred_idIdentitydnn/zero_fraction_3/LessEqual*
T0
*
_output_shapes
: 
?
,dnn/zero_fraction_3/cond/count_nonzero/zerosConst"^dnn/zero_fraction_3/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
/dnn/zero_fraction_3/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_3/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_3/cond/count_nonzero/zeros*
T0*'
_output_shapes
:?????????
?
6dnn/zero_fraction_3/cond/count_nonzero/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_3/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:?????????:?????????
?
+dnn/zero_fraction_3/cond/count_nonzero/CastCast/dnn/zero_fraction_3/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*'
_output_shapes
:?????????
?
,dnn/zero_fraction_3/cond/count_nonzero/ConstConst"^dnn/zero_fraction_3/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
4dnn/zero_fraction_3/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_3/cond/count_nonzero/Cast,dnn/zero_fraction_3/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
dnn/zero_fraction_3/cond/CastCast4dnn/zero_fraction_3/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
.dnn/zero_fraction_3/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_3/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
1dnn/zero_fraction_3/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_3/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:?????????
?
8dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_3/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:?????????:?????????
?
-dnn/zero_fraction_3/cond/count_nonzero_1/CastCast1dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*'
_output_shapes
:?????????
?
.dnn/zero_fraction_3/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_3/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
6dnn/zero_fraction_3/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_3/cond/count_nonzero_1/Cast.dnn/zero_fraction_3/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_3/cond/MergeMerge6dnn/zero_fraction_3/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_3/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
*dnn/zero_fraction_3/counts_to_fraction/subSubdnn/zero_fraction_3/Sizednn/zero_fraction_3/cond/Merge*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_3/counts_to_fraction/CastCast*dnn/zero_fraction_3/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_3/counts_to_fraction/Cast_1Castdnn/zero_fraction_3/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
.dnn/zero_fraction_3/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_3/counts_to_fraction/Cast-dnn/zero_fraction_3/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_3/fractionIdentity.dnn/zero_fraction_3/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values
?
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_3/fraction*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst*
_output_shapes
: *
dtype0**
value!B Bdnn/dnn/logits/activation
x
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
?
Clinear/linear_model/age_bucketized/weights/part_0/Initializer/zerosConst*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
?
1linear/linear_model/age_bucketized/weights/part_0VarHandleOp*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31linear/linear_model/age_bucketized/weights/part_0
?
Rlinear/linear_model/age_bucketized/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp1linear/linear_model/age_bucketized/weights/part_0*
_output_shapes
: 
?
8linear/linear_model/age_bucketized/weights/part_0/AssignAssignVariableOp1linear/linear_model/age_bucketized/weights/part_0Clinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros*
dtype0
?
Elinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0*
_output_shapes

:*
dtype0
?
\linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/Initializer/zerosConst*]
_classS
QOloc:@linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
?
Jlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0VarHandleOp*]
_classS
QOloc:@linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*[
shared_nameLJlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0
?
klinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpJlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0*
_output_shapes
: 
?
Qlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/AssignAssignVariableOpJlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0\linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/Initializer/zeros*
dtype0
?
^linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/Read/ReadVariableOpReadVariableOpJlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0*
_output_shapes

:*
dtype0
?
>linear/linear_model/education/weights/part_0/Initializer/zerosConst*?
_class5
31loc:@linear/linear_model/education/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
?
,linear/linear_model/education/weights/part_0VarHandleOp*?
_class5
31loc:@linear/linear_model/education/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,linear/linear_model/education/weights/part_0
?
Mlinear/linear_model/education/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/education/weights/part_0*
_output_shapes
: 
?
3linear/linear_model/education/weights/part_0/AssignAssignVariableOp,linear/linear_model/education/weights/part_0>linear/linear_model/education/weights/part_0/Initializer/zeros*
dtype0
?
@linear/linear_model/education/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/education/weights/part_0*
_output_shapes

:*
dtype0
?
Klinear/linear_model/education_X_occupation/weights/part_0/Initializer/zerosConst*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
?
9linear/linear_model/education_X_occupation/weights/part_0VarHandleOp*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9linear/linear_model/education_X_occupation/weights/part_0
?
Zlinear/linear_model/education_X_occupation/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp9linear/linear_model/education_X_occupation/weights/part_0*
_output_shapes
: 
?
@linear/linear_model/education_X_occupation/weights/part_0/AssignAssignVariableOp9linear/linear_model/education_X_occupation/weights/part_0Klinear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros*
dtype0
?
Mlinear/linear_model/education_X_occupation/weights/part_0/Read/ReadVariableOpReadVariableOp9linear/linear_model/education_X_occupation/weights/part_0*
_output_shapes

:*
dtype0
?
Clinear/linear_model/marital_status/weights/part_0/Initializer/zerosConst*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
?
1linear/linear_model/marital_status/weights/part_0VarHandleOp*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*B
shared_name31linear/linear_model/marital_status/weights/part_0
?
Rlinear/linear_model/marital_status/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp1linear/linear_model/marital_status/weights/part_0*
_output_shapes
: 
?
8linear/linear_model/marital_status/weights/part_0/AssignAssignVariableOp1linear/linear_model/marital_status/weights/part_0Clinear/linear_model/marital_status/weights/part_0/Initializer/zeros*
dtype0
?
Elinear/linear_model/marital_status/weights/part_0/Read/ReadVariableOpReadVariableOp1linear/linear_model/marital_status/weights/part_0*
_output_shapes

:*
dtype0
?
?linear/linear_model/occupation/weights/part_0/Initializer/zerosConst*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
?
-linear/linear_model/occupation/weights/part_0VarHandleOp*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*>
shared_name/-linear/linear_model/occupation/weights/part_0
?
Nlinear/linear_model/occupation/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/occupation/weights/part_0*
_output_shapes
: 
?
4linear/linear_model/occupation/weights/part_0/AssignAssignVariableOp-linear/linear_model/occupation/weights/part_0?linear/linear_model/occupation/weights/part_0/Initializer/zeros*
dtype0
?
Alinear/linear_model/occupation/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/occupation/weights/part_0*
_output_shapes

:*
dtype0
?
Alinear/linear_model/relationship/weights/part_0/Initializer/zerosConst*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
?
/linear/linear_model/relationship/weights/part_0VarHandleOp*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/linear/linear_model/relationship/weights/part_0
?
Plinear/linear_model/relationship/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp/linear/linear_model/relationship/weights/part_0*
_output_shapes
: 
?
6linear/linear_model/relationship/weights/part_0/AssignAssignVariableOp/linear/linear_model/relationship/weights/part_0Alinear/linear_model/relationship/weights/part_0/Initializer/zeros*
dtype0
?
Clinear/linear_model/relationship/weights/part_0/Read/ReadVariableOpReadVariableOp/linear/linear_model/relationship/weights/part_0*
_output_shapes

:*
dtype0
?
>linear/linear_model/workclass/weights/part_0/Initializer/zerosConst*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*
_output_shapes

:*
dtype0*
valueB*    
?
,linear/linear_model/workclass/weights/part_0VarHandleOp*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,linear/linear_model/workclass/weights/part_0
?
Mlinear/linear_model/workclass/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/workclass/weights/part_0*
_output_shapes
: 
?
3linear/linear_model/workclass/weights/part_0/AssignAssignVariableOp,linear/linear_model/workclass/weights/part_0>linear/linear_model/workclass/weights/part_0/Initializer/zeros*
dtype0
?
@linear/linear_model/workclass/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/workclass/weights/part_0*
_output_shapes

:*
dtype0
?
9linear/linear_model/bias_weights/part_0/Initializer/zerosConst*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0*
valueB*    
?
'linear/linear_model/bias_weights/part_0VarHandleOp*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'linear/linear_model/bias_weights/part_0
?
Hlinear/linear_model/bias_weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/bias_weights/part_0*
_output_shapes
: 
?
.linear/linear_model/bias_weights/part_0/AssignAssignVariableOp'linear/linear_model/bias_weights/part_09linear/linear_model/bias_weights/part_0/Initializer/zeros*
dtype0
?
;linear/linear_model/bias_weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0
?
Flinear/linear_model/linear_model/linear_model/age_bucketized/Bucketize	BucketizeParseExample/ParseExampleV2:15*
T0*'
_output_shapes
:?????????*:

boundaries,
*"(  ?A  ?A  ?A  B   B  4B  HB  \B  pB  ?B
?
Blinear/linear_model/linear_model/linear_model/age_bucketized/ShapeShapeFlinear/linear_model/linear_model/linear_model/age_bucketized/Bucketize*
T0*
_output_shapes
:
?
Plinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/strided_sliceStridedSliceBlinear/linear_model/linear_model/linear_model/age_bucketized/ShapePlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stackRlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stack_1Rlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Hlinear/linear_model/linear_model/linear_model/age_bucketized/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
Hlinear/linear_model/linear_model/linear_model/age_bucketized/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
Blinear/linear_model/linear_model/linear_model/age_bucketized/rangeRangeHlinear/linear_model/linear_model/linear_model/age_bucketized/range/startJlinear/linear_model/linear_model/linear_model/age_bucketized/strided_sliceHlinear/linear_model/linear_model/linear_model/age_bucketized/range/delta*#
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
?
Glinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims
ExpandDimsBlinear/linear_model/linear_model/linear_model/age_bucketized/rangeKlinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/age_bucketized/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
?
Alinear/linear_model/linear_model/linear_model/age_bucketized/TileTileGlinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDimsKlinear/linear_model/linear_model/linear_model/age_bucketized/Tile/multiples*
T0*'
_output_shapes
:?????????
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Dlinear/linear_model/linear_model/linear_model/age_bucketized/ReshapeReshapeAlinear/linear_model/linear_model/linear_model/age_bucketized/TileJlinear/linear_model/linear_model/linear_model/age_bucketized/Reshape/shape*
T0*#
_output_shapes
:?????????
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
Dlinear/linear_model/linear_model/linear_model/age_bucketized/range_1RangeJlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/startJlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/limitJlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/delta*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1/multiplesPackJlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice*
N*
T0*
_output_shapes
:
?
Clinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1TileDlinear/linear_model/linear_model/linear_model/age_bucketized/range_1Mlinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1/multiples*
T0*#
_output_shapes
:?????????
?
Llinear/linear_model/linear_model/linear_model/age_bucketized/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Flinear/linear_model/linear_model/linear_model/age_bucketized/Reshape_1ReshapeFlinear/linear_model/linear_model/linear_model/age_bucketized/BucketizeLlinear/linear_model/linear_model/linear_model/age_bucketized/Reshape_1/shape*
T0*#
_output_shapes
:?????????
?
Blinear/linear_model/linear_model/linear_model/age_bucketized/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
?
@linear/linear_model/linear_model/linear_model/age_bucketized/mulMulBlinear/linear_model/linear_model/linear_model/age_bucketized/mul/xClinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1*
T0*#
_output_shapes
:?????????
?
@linear/linear_model/linear_model/linear_model/age_bucketized/addAddV2Flinear/linear_model/linear_model/linear_model/age_bucketized/Reshape_1@linear/linear_model/linear_model/linear_model/age_bucketized/mul*
T0*#
_output_shapes
:?????????
?
Blinear/linear_model/linear_model/linear_model/age_bucketized/stackPackDlinear/linear_model/linear_model/linear_model/age_bucketized/ReshapeClinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1*
N*
T0*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/age_bucketized/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
?
Flinear/linear_model/linear_model/linear_model/age_bucketized/transpose	TransposeBlinear/linear_model/linear_model/linear_model/age_bucketized/stackKlinear/linear_model/linear_model/linear_model/age_bucketized/transpose/perm*
T0*'
_output_shapes
:?????????
?
Alinear/linear_model/linear_model/linear_model/age_bucketized/CastCastFlinear/linear_model/linear_model/linear_model/age_bucketized/transpose*

DstT0	*

SrcT0*'
_output_shapes
:?????????
?
Flinear/linear_model/linear_model/linear_model/age_bucketized/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B :
?
Dlinear/linear_model/linear_model/linear_model/age_bucketized/stack_1PackJlinear/linear_model/linear_model/linear_model/age_bucketized/strided_sliceFlinear/linear_model/linear_model/linear_model/age_bucketized/stack_1/1*
N*
T0*
_output_shapes
:
?
Clinear/linear_model/linear_model/linear_model/age_bucketized/Cast_1CastDlinear/linear_model/linear_model/linear_model/age_bucketized/stack_1*

DstT0	*

SrcT0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/age_bucketized/Shape_1/CastCastClinear/linear_model/linear_model/linear_model/age_bucketized/Cast_1*

DstT0*

SrcT0	*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Llinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1StridedSliceIlinear/linear_model/linear_model/linear_model/age_bucketized/Shape_1/CastRlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stackTlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stack_1Tlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Glinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2/x/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
Elinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2/xPackLlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1Glinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2/x/1*
N*
T0*
_output_shapes
:
?
Clinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2CastElinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2/x*

DstT0	*

SrcT0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshapeSparseReshapeAlinear/linear_model/linear_model/linear_model/age_bucketized/CastClinear/linear_model/linear_model/linear_model/age_bucketized/Cast_1Clinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2*-
_output_shapes
:?????????:
?
Slinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape/IdentityIdentity@linear/linear_model/linear_model/linear_model/age_bucketized/add*
T0*#
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SliceSliceLlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape:1Ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice/beginTlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Nlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ProdProdOlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SliceOlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Zlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2GatherV2Llinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape:1Zlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2/indicesWlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Cast/xPackNlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ProdRlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2*
N*
T0	*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshapeSparseReshapeJlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshapeLlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape:1Plinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
`linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshape/IdentityIdentitySlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape/Identity*
T0*#
_output_shapes
:?????????
?
Xlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GreaterEqualGreaterEqual`linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshape/IdentityXlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GreaterEqual/y*
T0*#
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/WhereWhereVlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ReshapeReshapeOlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/WhereWlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_1GatherV2Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshapeQlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_2GatherV2`linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshape/IdentityQlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/IdentityIdentityYlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
clinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
?
qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsTlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_1Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_2Rlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Identityclinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/Const*
T0*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceqlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stackwlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
hlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/UniqueUniqueslinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*2
_output_shapes 
:?????????:?????????
?
rlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather1linear/linear_model/age_bucketized/weights/part_0hlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*'
_output_shapes
:?????????*
dtype0
?
{linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityrlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*'
_output_shapes
:?????????
?
}linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity{linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparseSparseSegmentSum}linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1jlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/Unique:1olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
Slinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_1Reshapeslinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Ylinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ShapeShapealinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
]linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
_linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
_linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_sliceStridedSliceOlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Shape]linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_1_linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/stackPackQlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/stack/0Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/TileTileSlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_1Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/zeros_like	ZerosLikealinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Ilinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sumSelectNlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/TileTlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/zeros_likealinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Cast_1CastLlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1SlicePlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Cast_1Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1/beginVlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Shape_1ShapeIlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum*
T0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2SliceQlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Shape_1Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2/beginVlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Plinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/concatConcatV2Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2Ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/concat/axis*
N*
T0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_2ReshapeIlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sumPlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
[linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/ShapeShapeFlinear/linear_model/linear_model/linear_model/age_bucketized/Bucketize*
T0*
_output_shapes
:
?
ilinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_sliceStridedSlice[linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Shapeilinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice/stackklinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice/stack_1klinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
alinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
alinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
[linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/rangeRangealinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range/startclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slicealinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range/delta*#
_output_shapes
:?????????
?
dlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
?
`linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/ExpandDims
ExpandDims[linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/rangedlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
dlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      
?
Zlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/TileTile`linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/ExpandDimsdlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Tile/multiples*
T0*'
_output_shapes
:?????????
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
]linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/ReshapeReshapeZlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Tileclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Reshape/shape*
T0*#
_output_shapes
:?????????
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
]linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range_1Rangeclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range_1/startclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range_1/limitclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range_1/delta*
_output_shapes
:
?
flinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Tile_1/multiplesPackclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice*
N*
T0*
_output_shapes
:
?
\linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Tile_1Tile]linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/range_1flinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Tile_1/multiples*
T0*#
_output_shapes
:?????????
?
elinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
_linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Reshape_1ReshapeFlinear/linear_model/linear_model/linear_model/age_bucketized/Bucketizeelinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Reshape_1/shape*
T0*#
_output_shapes
:?????????
?
[linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/mul/xConst*
_output_shapes
: *
dtype0*
value	B :
?
Ylinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/mulMul[linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/mul/x\linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Tile_1*
T0*#
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/addAddV2_linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Reshape_1Ylinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/mul*
T0*#
_output_shapes
:?????????
?
[linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/stackPack]linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Reshape\linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Tile_1*
N*
T0*'
_output_shapes
:?????????
?
dlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
?
_linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/transpose	Transpose[linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/stackdlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/transpose/perm*
T0*'
_output_shapes
:?????????
?
Zlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/CastCast_linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/transpose*

DstT0	*

SrcT0*'
_output_shapes
:?????????
?
_linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B :
?
]linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/stack_1Packclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice_linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/stack_1/1*
N*
T0*
_output_shapes
:
?
\linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_1Cast]linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/stack_1*

DstT0	*

SrcT0*
_output_shapes
:
?
\linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_2CastYlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/add*

DstT0	*

SrcT0*#
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseCrossSparseCrossZlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/CastParseExample/ParseExampleV2ParseExample/ParseExampleV2:2\linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_2ParseExample/ParseExampleV2:5ParseExample/ParseExampleV2:7\linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_1ParseExample/ParseExampleV2:10ParseExample/ParseExampleV2:12*
N*<
_output_shapes*
(:?????????:?????????:*
dense_types
 *
hash_key?????*
hashed_output(*
internal_type0	*
num_buckets*
out_type0	*
sparse_types
2	
?
blinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Shape_1/CastCastclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseCross:2*

DstT0*

SrcT0	*
_output_shapes
:
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
elinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice_1StridedSliceblinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Shape_1/Castklinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice_1/stackmlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice_1/stack_1mlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
`linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_3/x/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
^linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_3/xPackelinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/strided_slice_1`linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_3/x/1*
N*
T0*
_output_shapes
:
?
\linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_3Cast^linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_3/x*

DstT0	*

SrcT0*
_output_shapes
:
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseReshapeSparseReshapealinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseCrossclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseCross:2\linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/Cast_3*-
_output_shapes
:?????????:
?
llinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseReshape/IdentityIdentityclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseCross:1*
T0	*#
_output_shapes
:?????????
?
nlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SliceSliceelinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseReshape:1nlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice/beginmlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
glinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/ProdProdhlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slicehlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Const*
T0	*
_output_shapes
: 
?
slinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2GatherV2elinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseReshape:1slinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2/indicesplinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
ilinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Cast/xPackglinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Prodklinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2*
N*
T0	*
_output_shapes
:
?
plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseReshapeSparseReshapeclinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseReshapeelinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseReshape:1ilinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
ylinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseReshape/IdentityIdentityllinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
qlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
olinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GreaterEqualGreaterEqualylinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseReshape/Identityqlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/WhereWhereolinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
jlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/ReshapeReshapehlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Whereplinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
rlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2_1GatherV2plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseReshapejlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshaperlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
rlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2_2GatherV2ylinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseReshape/Identityjlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshaperlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/IdentityIdentityrlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
|linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsmlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2_1mlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/GatherV2_2klinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Identity|linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlice?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/UniqueUnique?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGatherJlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*]
_classS
QOloc:@linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0*'
_output_shapes
:?????????*
dtype0
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*]
_classS
QOloc:@linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0*'
_output_shapes
:?????????
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
zlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparseSparseSegmentSum?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/Unique:1?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
rlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
llinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshape_1Reshape?linear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2rlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/ShapeShapezlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
vlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
xlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
xlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/strided_sliceStridedSlicehlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Shapevlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/strided_slice/stackxlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/strided_slice/stack_1xlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
jlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/stackPackjlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/stack/0plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:
?
glinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/TileTilellinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshape_1hlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/zeros_like	ZerosLikezlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
blinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sumSelectglinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Tilemlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/zeros_likezlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
ilinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Cast_1Castelinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
?
plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
olinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
jlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_1Sliceilinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Cast_1plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_1/beginolinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
jlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Shape_1Shapeblinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum*
T0*
_output_shapes
:
?
plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
olinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
jlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_2Slicejlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Shape_1plinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_2/beginolinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
nlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
ilinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/concatConcatV2jlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_1jlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Slice_2nlinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/concat/axis*
N*
T0*
_output_shapes
:
?
llinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshape_2Reshapeblinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sumilinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Nlinear/linear_model/linear_model/linear_model/education/education_lookup/ConstConst*
_output_shapes
:*
dtype0*?
value?B?B	BachelorsBHS-gradB11thBMastersB9thBSome-collegeB
Assoc-acdmB	Assoc-vocB7th-8thB	DoctorateBProf-schoolB5th-6thB10thB1st-4thB	PreschoolB12th
?
Mlinear/linear_model/linear_model/linear_model/education/education_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
Tlinear/linear_model/linear_model/linear_model/education/education_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
Tlinear/linear_model/linear_model/linear_model/education/education_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
Nlinear/linear_model/linear_model/linear_model/education/education_lookup/rangeRangeTlinear/linear_model/linear_model/linear_model/education/education_lookup/range/startMlinear/linear_model/linear_model/linear_model/education/education_lookup/SizeTlinear/linear_model/linear_model/linear_model/education/education_lookup/range/delta*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/education/education_lookup/CastCastNlinear/linear_model/linear_model/linear_model/education/education_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
Ylinear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
^linear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_62ab4038-d62e-4579-bd6a-37d063900a1e*
value_dtype0	
?
rlinear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2^linear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/hash_tableNlinear/linear_model/linear_model/linear_model/education/education_lookup/ConstMlinear/linear_model/linear_model/linear_model/education/education_lookup/Cast*	
Tin0*

Tout0	
?
[linear/linear_model/linear_model/linear_model/education/hash_table_Lookup/LookupTableFindV2LookupTableFindV2^linear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/hash_tableParseExample/ParseExampleV2:5Ylinear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
Blinear/linear_model/linear_model/linear_model/education/Shape/CastCastParseExample/ParseExampleV2:10*

DstT0*

SrcT0	*
_output_shapes
:
?
Klinear/linear_model/linear_model/linear_model/education/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Mlinear/linear_model/linear_model/linear_model/education/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Mlinear/linear_model/linear_model/linear_model/education/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Elinear/linear_model/linear_model/linear_model/education/strided_sliceStridedSliceBlinear/linear_model/linear_model/linear_model/education/Shape/CastKlinear/linear_model/linear_model/linear_model/education/strided_slice/stackMlinear/linear_model/linear_model/linear_model/education/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/education/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
@linear/linear_model/linear_model/linear_model/education/Cast/x/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
>linear/linear_model/linear_model/linear_model/education/Cast/xPackElinear/linear_model/linear_model/linear_model/education/strided_slice@linear/linear_model/linear_model/linear_model/education/Cast/x/1*
N*
T0*
_output_shapes
:
?
<linear/linear_model/linear_model/linear_model/education/CastCast>linear/linear_model/linear_model/linear_model/education/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
?
Elinear/linear_model/linear_model/linear_model/education/SparseReshapeSparseReshapeParseExample/ParseExampleV2ParseExample/ParseExampleV2:10<linear/linear_model/linear_model/linear_model/education/Cast*-
_output_shapes
:?????????:
?
Nlinear/linear_model/linear_model/linear_model/education/SparseReshape/IdentityIdentity[linear/linear_model/linear_model/linear_model/education/hash_table_Lookup/LookupTableFindV2*
T0	*#
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Olinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/SliceSliceGlinear/linear_model/linear_model/linear_model/education/SparseReshape:1Plinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice/beginOlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ilinear/linear_model/linear_model/linear_model/education/weighted_sum/ProdProdJlinear/linear_model/linear_model/linear_model/education/weighted_sum/SliceJlinear/linear_model/linear_model/linear_model/education/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Ulinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Mlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2GatherV2Glinear/linear_model/linear_model/linear_model/education/SparseReshape:1Ulinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2/indicesRlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
Klinear/linear_model/linear_model/linear_model/education/weighted_sum/Cast/xPackIlinear/linear_model/linear_model/linear_model/education/weighted_sum/ProdMlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2*
N*
T0	*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshapeSparseReshapeElinear/linear_model/linear_model/linear_model/education/SparseReshapeGlinear/linear_model/linear_model/linear_model/education/SparseReshape:1Klinear/linear_model/linear_model/linear_model/education/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
[linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshape/IdentityIdentityNlinear/linear_model/linear_model/linear_model/education/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/education/weighted_sum/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Qlinear/linear_model/linear_model/linear_model/education/weighted_sum/GreaterEqualGreaterEqual[linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshape/IdentitySlinear/linear_model/linear_model/linear_model/education/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/WhereWhereQlinear/linear_model/linear_model/linear_model/education/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/ReshapeReshapeJlinear/linear_model/linear_model/linear_model/education/weighted_sum/WhereRlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Olinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_1GatherV2Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshapeLlinear/linear_model/linear_model/linear_model/education/weighted_sum/ReshapeTlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Olinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_2GatherV2[linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshape/IdentityLlinear/linear_model/linear_model/linear_model/education/weighted_sum/ReshapeTlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
Mlinear/linear_model/linear_model/linear_model/education/weighted_sum/IdentityIdentityTlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
^linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
llinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsOlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_1Olinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_2Mlinear/linear_model/linear_model/linear_model/education/weighted_sum/Identity^linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
plinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
rlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
rlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
jlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlicellinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsplinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stackrlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1rlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
clinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/UniqueUniquenlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
mlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather,linear/linear_model/education/weights/part_0clinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*?
_class5
31loc:@linear/linear_model/education/weights/part_0*'
_output_shapes
:?????????*
dtype0
?
vlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentitymlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*?
_class5
31loc:@linear/linear_model/education/weights/part_0*'
_output_shapes
:?????????
?
xlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identityvlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
\linear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparseSparseSegmentSumxlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1elinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/Unique:1jlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
Nlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_1Reshapenlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Tlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/ShapeShape\linear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
Xlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
Zlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Zlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_sliceStridedSliceJlinear/linear_model/linear_model/linear_model/education/weighted_sum/ShapeXlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stackZlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stack_1Zlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/stackPackLlinear/linear_model/linear_model/linear_model/education/weighted_sum/stack/0Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/education/weighted_sum/TileTileNlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_1Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Olinear/linear_model/linear_model/linear_model/education/weighted_sum/zeros_like	ZerosLike\linear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Dlinear/linear_model/linear_model/linear_model/education/weighted_sumSelectIlinear/linear_model/linear_model/linear_model/education/weighted_sum/TileOlinear/linear_model/linear_model/linear_model/education/weighted_sum/zeros_like\linear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/education/weighted_sum/Cast_1CastGlinear/linear_model/linear_model/linear_model/education/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Qlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1SliceKlinear/linear_model/linear_model/linear_model/education/weighted_sum/Cast_1Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1/beginQlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Shape_1ShapeDlinear/linear_model/linear_model/linear_model/education/weighted_sum*
T0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
Qlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2SliceLlinear/linear_model/linear_model/linear_model/education/weighted_sum/Shape_1Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2/beginQlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Plinear/linear_model/linear_model/linear_model/education/weighted_sum/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Klinear/linear_model/linear_model/linear_model/education/weighted_sum/concatConcatV2Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2Plinear/linear_model/linear_model/linear_model/education/weighted_sum/concat/axis*
N*
T0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_2ReshapeDlinear/linear_model/linear_model/linear_model/education/weighted_sumKlinear/linear_model/linear_model/linear_model/education/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCrossSparseCrossParseExample/ParseExampleV2ParseExample/ParseExampleV2:2ParseExample/ParseExampleV2:5ParseExample/ParseExampleV2:7ParseExample/ParseExampleV2:10ParseExample/ParseExampleV2:12*
N*<
_output_shapes*
(:?????????:?????????:*
dense_types
 *
hash_key?????*
hashed_output(*
internal_type0*
num_buckets*
out_type0	*
sparse_types
2
?
Olinear/linear_model/linear_model/linear_model/education_X_occupation/Shape/CastCastRlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCross:2*

DstT0*

SrcT0	*
_output_shapes
:
?
Xlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Zlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Zlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Rlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_sliceStridedSliceOlinear/linear_model/linear_model/linear_model/education_X_occupation/Shape/CastXlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stackZlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stack_1Zlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Mlinear/linear_model/linear_model/linear_model/education_X_occupation/Cast/x/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
Klinear/linear_model/linear_model/linear_model/education_X_occupation/Cast/xPackRlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_sliceMlinear/linear_model/linear_model/linear_model/education_X_occupation/Cast/x/1*
N*
T0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/education_X_occupation/CastCastKlinear/linear_model/linear_model/linear_model/education_X_occupation/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshapeSparseReshapePlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCrossRlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCross:2Ilinear/linear_model/linear_model/linear_model/education_X_occupation/Cast*-
_output_shapes
:?????????:
?
[linear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape/IdentityIdentityRlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCross:1*
T0	*#
_output_shapes
:?????????
?
]linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SliceSliceTlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape:1]linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice/begin\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Vlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ProdProdWlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SliceWlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Const*
T0	*
_output_shapes
: 
?
blinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Zlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2GatherV2Tlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape:1blinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2/indices_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
Xlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Cast/xPackVlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ProdZlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2*
N*
T0	*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshapeSparseReshapeRlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshapeTlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape:1Xlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
hlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshape/IdentityIdentity[linear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
`linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GreaterEqualGreaterEqualhlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshape/Identity`linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/WhereWhere^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ReshapeReshapeWlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Where_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_1GatherV2_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshapeYlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshapealinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_2GatherV2hlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshape/IdentityYlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshapealinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
Zlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/IdentityIdentityalinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
klinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_1\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_2Zlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Identityklinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
}linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows}linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stacklinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
plinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/UniqueUnique{linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
zlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather9linear/linear_model/education_X_occupation/weights/part_0plinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*'
_output_shapes
:?????????*
dtype0
?
?linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityzlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*'
_output_shapes
:?????????
?
?linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
ilinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparseSparseSegmentSum?linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1rlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/Unique:1wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
[linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_1Reshape{linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2alinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ShapeShapeilinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
elinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
glinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
glinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_sliceStridedSliceWlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Shapeelinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stackglinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stack_1glinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/stackPackYlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/stack/0_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:
?
Vlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/TileTile[linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_1Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/zeros_like	ZerosLikeilinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Qlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sumSelectVlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Tile\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/zeros_likeilinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Xlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Cast_1CastTlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1SliceXlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Cast_1_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1/begin^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Shape_1ShapeQlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum*
T0*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2SliceYlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Shape_1_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2/begin^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
]linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Xlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/concatConcatV2Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2]linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/concat/axis*
N*
T0*
_output_shapes
:
?
[linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_2ReshapeQlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sumXlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Xlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/ConstConst*
_output_shapes
:*
dtype0*~
valueuBsBMarried-civ-spouseBDivorcedBMarried-spouse-absentBNever-marriedB	SeparatedBMarried-AF-spouseBWidowed
?
Wlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
Xlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/rangeRange^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range/startWlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/Size^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range/delta*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/CastCastXlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
clinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
hlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_f70cf81a-91f4-40fd-892e-64dbc47d1a28*
value_dtype0	
?
|linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2hlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/hash_tableXlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/ConstWlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/Cast*	
Tin0*

Tout0	
?
`linear/linear_model/linear_model/linear_model/marital_status/hash_table_Lookup/LookupTableFindV2LookupTableFindV2hlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/hash_tableParseExample/ParseExampleV2:6clinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
Glinear/linear_model/linear_model/linear_model/marital_status/Shape/CastCastParseExample/ParseExampleV2:11*

DstT0*

SrcT0	*
_output_shapes
:
?
Plinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Rlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Rlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Jlinear/linear_model/linear_model/linear_model/marital_status/strided_sliceStridedSliceGlinear/linear_model/linear_model/linear_model/marital_status/Shape/CastPlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stackRlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stack_1Rlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Elinear/linear_model/linear_model/linear_model/marital_status/Cast/x/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
Clinear/linear_model/linear_model/linear_model/marital_status/Cast/xPackJlinear/linear_model/linear_model/linear_model/marital_status/strided_sliceElinear/linear_model/linear_model/linear_model/marital_status/Cast/x/1*
N*
T0*
_output_shapes
:
?
Alinear/linear_model/linear_model/linear_model/marital_status/CastCastClinear/linear_model/linear_model/linear_model/marital_status/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/marital_status/SparseReshapeSparseReshapeParseExample/ParseExampleV2:1ParseExample/ParseExampleV2:11Alinear/linear_model/linear_model/linear_model/marital_status/Cast*-
_output_shapes
:?????????:
?
Slinear/linear_model/linear_model/linear_model/marital_status/SparseReshape/IdentityIdentity`linear/linear_model/linear_model/linear_model/marital_status/hash_table_Lookup/LookupTableFindV2*
T0	*#
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SliceSliceLlinear/linear_model/linear_model/linear_model/marital_status/SparseReshape:1Ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice/beginTlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Nlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ProdProdOlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SliceOlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Zlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Rlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2GatherV2Llinear/linear_model/linear_model/linear_model/marital_status/SparseReshape:1Zlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2/indicesWlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Cast/xPackNlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ProdRlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2*
N*
T0	*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshapeSparseReshapeJlinear/linear_model/linear_model/linear_model/marital_status/SparseReshapeLlinear/linear_model/linear_model/linear_model/marital_status/SparseReshape:1Plinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
`linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshape/IdentityIdentitySlinear/linear_model/linear_model/linear_model/marital_status/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Xlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Vlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GreaterEqualGreaterEqual`linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshape/IdentityXlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/WhereWhereVlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ReshapeReshapeOlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/WhereWlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_1GatherV2Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshapeQlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_2GatherV2`linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshape/IdentityQlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
Rlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/IdentityIdentityYlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
clinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsTlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_1Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_2Rlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Identityclinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceqlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stackwlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
hlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/UniqueUniqueslinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
rlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather1linear/linear_model/marital_status/weights/part_0hlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*'
_output_shapes
:?????????*
dtype0
?
{linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityrlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*'
_output_shapes
:?????????
?
}linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity{linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparseSparseSegmentSum}linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1jlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/Unique:1olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
Slinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_1Reshapeslinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Ylinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ShapeShapealinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
]linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
_linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
_linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_sliceStridedSliceOlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Shape]linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_1_linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/stackPackQlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/stack/0Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/TileTileSlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_1Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/zeros_like	ZerosLikealinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Ilinear/linear_model/linear_model/linear_model/marital_status/weighted_sumSelectNlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/TileTlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/zeros_likealinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Cast_1CastLlinear/linear_model/linear_model/linear_model/marital_status/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Vlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1SlicePlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Cast_1Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1/beginVlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Shape_1ShapeIlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum*
T0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
Vlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2SliceQlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Shape_1Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2/beginVlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Plinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/concatConcatV2Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2Ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/concat/axis*
N*
T0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_2ReshapeIlinear/linear_model/linear_model/linear_model/marital_status/weighted_sumPlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
?linear/linear_model/linear_model/linear_model/occupation/lookupStringToHashBucketFastParseExample/ParseExampleV2:7*#
_output_shapes
:?????????*
num_buckets
?
Clinear/linear_model/linear_model/linear_model/occupation/Shape/CastCastParseExample/ParseExampleV2:12*

DstT0*

SrcT0	*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/occupation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Nlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Nlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Flinear/linear_model/linear_model/linear_model/occupation/strided_sliceStridedSliceClinear/linear_model/linear_model/linear_model/occupation/Shape/CastLlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stackNlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Alinear/linear_model/linear_model/linear_model/occupation/Cast/x/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
?linear/linear_model/linear_model/linear_model/occupation/Cast/xPackFlinear/linear_model/linear_model/linear_model/occupation/strided_sliceAlinear/linear_model/linear_model/linear_model/occupation/Cast/x/1*
N*
T0*
_output_shapes
:
?
=linear/linear_model/linear_model/linear_model/occupation/CastCast?linear/linear_model/linear_model/linear_model/occupation/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
?
Flinear/linear_model/linear_model/linear_model/occupation/SparseReshapeSparseReshapeParseExample/ParseExampleV2:2ParseExample/ParseExampleV2:12=linear/linear_model/linear_model/linear_model/occupation/Cast*-
_output_shapes
:?????????:
?
Olinear/linear_model/linear_model/linear_model/occupation/SparseReshape/IdentityIdentity?linear/linear_model/linear_model/linear_model/occupation/lookup*
T0	*#
_output_shapes
:?????????
?
Qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SliceSliceHlinear/linear_model/linear_model/linear_model/occupation/SparseReshape:1Qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice/beginPlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Jlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ProdProdKlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SliceKlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Vlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Nlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2GatherV2Hlinear/linear_model/linear_model/linear_model/occupation/SparseReshape:1Vlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2/indicesSlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
Llinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Cast/xPackJlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ProdNlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2*
N*
T0	*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshapeSparseReshapeFlinear/linear_model/linear_model/linear_model/occupation/SparseReshapeHlinear/linear_model/linear_model/linear_model/occupation/SparseReshape:1Llinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
\linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshape/IdentityIdentityOlinear/linear_model/linear_model/linear_model/occupation/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Rlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GreaterEqualGreaterEqual\linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshape/IdentityTlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/WhereWhereRlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ReshapeReshapeKlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/WhereSlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_1GatherV2Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshapeMlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ReshapeUlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_2GatherV2\linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshape/IdentityMlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ReshapeUlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
Nlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/IdentityIdentityUlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsPlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_1Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_2Nlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Identity_linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlicemlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsqlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stackslinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
dlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/UniqueUniqueolinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
nlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather-linear/linear_model/occupation/weights/part_0dlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*'
_output_shapes
:?????????*
dtype0
?
wlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentitynlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*'
_output_shapes
:?????????
?
ylinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identitywlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
]linear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparseSparseSegmentSumylinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1flinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/Unique:1klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
Olinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_1Reshapeolinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Ulinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ShapeShape]linear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
Ylinear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
[linear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
[linear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_sliceStridedSliceKlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ShapeYlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack[linear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack_1[linear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/stackPackMlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/stack/0Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/TileTileOlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_1Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/zeros_like	ZerosLike]linear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Elinear/linear_model/linear_model/linear_model/occupation/weighted_sumSelectJlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/TilePlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/zeros_like]linear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Llinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Cast_1CastHlinear/linear_model/linear_model/linear_model/occupation/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Rlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1SliceLlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Cast_1Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1/beginRlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Shape_1ShapeElinear/linear_model/linear_model/linear_model/occupation/weighted_sum*
T0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
Rlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2SliceMlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Shape_1Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2/beginRlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Llinear/linear_model/linear_model/linear_model/occupation/weighted_sum/concatConcatV2Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2Qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/concat/axis*
N*
T0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_2ReshapeElinear/linear_model/linear_model/linear_model/occupation/weighted_sumLlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/ConstConst*
_output_shapes
:*
dtype0*W
valueNBLBHusbandBNot-in-familyBWifeB	Own-childB	UnmarriedBOther-relative
?
Slinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
?
Zlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
?
Zlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
Tlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/rangeRangeZlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range/startSlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/SizeZlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range/delta*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/CastCastTlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
dlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*@
shared_name1/hash_table_3f1adf99-c6c3-4506-baff-821f791a02d5*
value_dtype0	
?
xlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2dlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/hash_tableTlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/ConstSlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/Cast*	
Tin0*

Tout0	
?
^linear/linear_model/linear_model/linear_model/relationship/hash_table_Lookup/LookupTableFindV2LookupTableFindV2dlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/hash_tableParseExample/ParseExampleV2:8_linear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
Elinear/linear_model/linear_model/linear_model/relationship/Shape/CastCastParseExample/ParseExampleV2:13*

DstT0*

SrcT0	*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/relationship/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Plinear/linear_model/linear_model/linear_model/relationship/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Plinear/linear_model/linear_model/linear_model/relationship/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Hlinear/linear_model/linear_model/linear_model/relationship/strided_sliceStridedSliceElinear/linear_model/linear_model/linear_model/relationship/Shape/CastNlinear/linear_model/linear_model/linear_model/relationship/strided_slice/stackPlinear/linear_model/linear_model/linear_model/relationship/strided_slice/stack_1Plinear/linear_model/linear_model/linear_model/relationship/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Clinear/linear_model/linear_model/linear_model/relationship/Cast/x/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
Alinear/linear_model/linear_model/linear_model/relationship/Cast/xPackHlinear/linear_model/linear_model/linear_model/relationship/strided_sliceClinear/linear_model/linear_model/linear_model/relationship/Cast/x/1*
N*
T0*
_output_shapes
:
?
?linear/linear_model/linear_model/linear_model/relationship/CastCastAlinear/linear_model/linear_model/linear_model/relationship/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
?
Hlinear/linear_model/linear_model/linear_model/relationship/SparseReshapeSparseReshapeParseExample/ParseExampleV2:3ParseExample/ParseExampleV2:13?linear/linear_model/linear_model/linear_model/relationship/Cast*-
_output_shapes
:?????????:
?
Qlinear/linear_model/linear_model/linear_model/relationship/SparseReshape/IdentityIdentity^linear/linear_model/linear_model/linear_model/relationship/hash_table_Lookup/LookupTableFindV2*
T0	*#
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SliceSliceJlinear/linear_model/linear_model/linear_model/relationship/SparseReshape:1Slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice/beginRlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Llinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ProdProdMlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SliceMlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Xlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Plinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2GatherV2Jlinear/linear_model/linear_model/linear_model/relationship/SparseReshape:1Xlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2/indicesUlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
Nlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Cast/xPackLlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ProdPlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2*
N*
T0	*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshapeSparseReshapeHlinear/linear_model/linear_model/linear_model/relationship/SparseReshapeJlinear/linear_model/linear_model/linear_model/relationship/SparseReshape:1Nlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
^linear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshape/IdentityIdentityQlinear/linear_model/linear_model/linear_model/relationship/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Vlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Tlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GreaterEqualGreaterEqual^linear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshape/IdentityVlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/WhereWhereTlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ReshapeReshapeMlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/WhereUlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_1GatherV2Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshapeOlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ReshapeWlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_2GatherV2^linear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshape/IdentityOlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ReshapeWlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/relationship/weighted_sum/IdentityIdentityWlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
alinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsRlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_1Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_2Plinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Identityalinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceolinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsslinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stackulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
flinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/UniqueUniqueqlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
plinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather/linear/linear_model/relationship/weights/part_0flinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*'
_output_shapes
:?????????*
dtype0
?
ylinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityplinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*'
_output_shapes
:?????????
?
{linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identityylinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
_linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparseSparseSegmentSum{linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1hlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/Unique:1mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
Qlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_1Reshapeqlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Wlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ShapeShape_linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
[linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
]linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
]linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_sliceStridedSliceMlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Shape[linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack]linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack_1]linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/stackPackOlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/stack/0Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/relationship/weighted_sum/TileTileQlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_1Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/zeros_like	ZerosLike_linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Glinear/linear_model/linear_model/linear_model/relationship/weighted_sumSelectLlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/TileRlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/zeros_like_linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Nlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Cast_1CastJlinear/linear_model/linear_model/linear_model/relationship/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Tlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1SliceNlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Cast_1Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1/beginTlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Shape_1ShapeGlinear/linear_model/linear_model/linear_model/relationship/weighted_sum*
T0*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
Tlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2SliceOlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Shape_1Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2/beginTlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Nlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/concatConcatV2Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2Slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/concat/axis*
N*
T0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_2ReshapeGlinear/linear_model/linear_model/linear_model/relationship/weighted_sumNlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
>linear/linear_model/linear_model/linear_model/workclass/lookupStringToHashBucketFastParseExample/ParseExampleV2:9*#
_output_shapes
:?????????*
num_buckets
?
Blinear/linear_model/linear_model/linear_model/workclass/Shape/CastCastParseExample/ParseExampleV2:14*

DstT0*

SrcT0	*
_output_shapes
:
?
Klinear/linear_model/linear_model/linear_model/workclass/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?
Mlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Mlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Elinear/linear_model/linear_model/linear_model/workclass/strided_sliceStridedSliceBlinear/linear_model/linear_model/linear_model/workclass/Shape/CastKlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stackMlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
@linear/linear_model/linear_model/linear_model/workclass/Cast/x/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????
?
>linear/linear_model/linear_model/linear_model/workclass/Cast/xPackElinear/linear_model/linear_model/linear_model/workclass/strided_slice@linear/linear_model/linear_model/linear_model/workclass/Cast/x/1*
N*
T0*
_output_shapes
:
?
<linear/linear_model/linear_model/linear_model/workclass/CastCast>linear/linear_model/linear_model/linear_model/workclass/Cast/x*

DstT0	*

SrcT0*
_output_shapes
:
?
Elinear/linear_model/linear_model/linear_model/workclass/SparseReshapeSparseReshapeParseExample/ParseExampleV2:4ParseExample/ParseExampleV2:14<linear/linear_model/linear_model/linear_model/workclass/Cast*-
_output_shapes
:?????????:
?
Nlinear/linear_model/linear_model/linear_model/workclass/SparseReshape/IdentityIdentity>linear/linear_model/linear_model/linear_model/workclass/lookup*
T0	*#
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SliceSliceGlinear/linear_model/linear_model/linear_model/workclass/SparseReshape:1Plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice/beginOlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ilinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ProdProdJlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SliceJlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Ulinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Mlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2GatherV2Glinear/linear_model/linear_model/linear_model/workclass/SparseReshape:1Ulinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2/indicesRlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
Klinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Cast/xPackIlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ProdMlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2*
N*
T0	*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshapeSparseReshapeElinear/linear_model/linear_model/linear_model/workclass/SparseReshapeGlinear/linear_model/linear_model/linear_model/workclass/SparseReshape:1Klinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
[linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshape/IdentityIdentityNlinear/linear_model/linear_model/linear_model/workclass/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Qlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GreaterEqualGreaterEqual[linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshape/IdentitySlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/WhereWhereQlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ReshapeReshapeJlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/WhereRlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_1GatherV2Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshapeLlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ReshapeTlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_2GatherV2[linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshape/IdentityLlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ReshapeTlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
Mlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/IdentityIdentityTlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
^linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsOlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_1Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_2Mlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Identity^linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
?
rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
?
rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
?
jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlicellinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsplinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stackrlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask
?
clinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/UniqueUniquenlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
mlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather,linear/linear_model/workclass/weights/part_0clinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*'
_output_shapes
:?????????*
dtype0
?
vlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentitymlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*'
_output_shapes
:?????????
?
xlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identityvlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
\linear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparseSparseSegmentSumxlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1elinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/Unique:1jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice*
T0*
Tsegmentids0	*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   
?
Nlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_1Reshapenlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Tlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ShapeShape\linear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
Xlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
Zlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Zlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_sliceStridedSliceJlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ShapeXlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stackZlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stack_1Zlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/stack/0Const*
_output_shapes
: *
dtype0*
value	B :
?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/stackPackLlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/stack/0Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice*
N*
T0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/workclass/weighted_sum/TileTileNlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_1Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/zeros_like	ZerosLike\linear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Dlinear/linear_model/linear_model/linear_model/workclass/weighted_sumSelectIlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/TileOlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/zeros_like\linear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Cast_1CastGlinear/linear_model/linear_model/linear_model/workclass/SparseReshape:1*

DstT0*

SrcT0	*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: 
?
Qlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1SliceKlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Cast_1Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1/beginQlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Shape_1ShapeDlinear/linear_model/linear_model/linear_model/workclass/weighted_sum*
T0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:
?
Qlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2SliceLlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Shape_1Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2/beginQlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Klinear/linear_model/linear_model/linear_model/workclass/weighted_sum/concatConcatV2Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2Plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/concat/axis*
N*
T0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_2ReshapeDlinear/linear_model/linear_model/linear_model/workclass/weighted_sumKlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Blinear/linear_model/linear_model/linear_model/weighted_sum_no_biasAddNSlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_2llinear/linear_model/linear_model/linear_model/age_bucketized_X_education_X_occupation/weighted_sum/Reshape_2Nlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_2[linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_2Slinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_2Olinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_2Qlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_2Nlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_2*
N*
T0*'
_output_shapes
:?????????
?
/linear/linear_model/bias_weights/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0
?
 linear/linear_model/bias_weightsIdentity/linear/linear_model/bias_weights/ReadVariableOp*
T0*
_output_shapes
:
?
:linear/linear_model/linear_model/linear_model/weighted_sumBiasAddBlinear/linear_model/linear_model/linear_model/weighted_sum_no_bias linear/linear_model/bias_weights*
T0*'
_output_shapes
:?????????
y
linear/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
_output_shapes
:*
dtype0
d
linear/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
f
linear/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
f
linear/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
linear/strided_sliceStridedSlicelinear/ReadVariableOplinear/strided_slice/stacklinear/strided_slice/stack_1linear/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
\
linear/bias/tagsConst*
_output_shapes
: *
dtype0*
valueB Blinear/bias
e
linear/biasScalarSummarylinear/bias/tagslinear/strided_slice*
T0*
_output_shapes
: 
?
3linear/zero_fraction/total_size/Size/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0*
_output_shapes

:*
dtype0
f
$linear/zero_fraction/total_size/SizeConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
5linear/zero_fraction/total_size/Size_1/ReadVariableOpReadVariableOpJlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
5linear/zero_fraction/total_size/Size_2/ReadVariableOpReadVariableOp,linear/linear_model/education/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_2Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
5linear/zero_fraction/total_size/Size_3/ReadVariableOpReadVariableOp9linear/linear_model/education_X_occupation/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_3Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
5linear/zero_fraction/total_size/Size_4/ReadVariableOpReadVariableOp1linear/linear_model/marital_status/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_4Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
5linear/zero_fraction/total_size/Size_5/ReadVariableOpReadVariableOp-linear/linear_model/occupation/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_5Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
5linear/zero_fraction/total_size/Size_6/ReadVariableOpReadVariableOp/linear/linear_model/relationship/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_6Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
5linear/zero_fraction/total_size/Size_7/ReadVariableOpReadVariableOp,linear/linear_model/workclass/weights/part_0*
_output_shapes

:*
dtype0
h
&linear/zero_fraction/total_size/Size_7Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
$linear/zero_fraction/total_size/AddNAddN$linear/zero_fraction/total_size/Size&linear/zero_fraction/total_size/Size_1&linear/zero_fraction/total_size/Size_2&linear/zero_fraction/total_size/Size_3&linear/zero_fraction/total_size/Size_4&linear/zero_fraction/total_size/Size_5&linear/zero_fraction/total_size/Size_6&linear/zero_fraction/total_size/Size_7*
N*
T0	*
_output_shapes
: 
g
%linear/zero_fraction/total_zero/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
%linear/zero_fraction/total_zero/EqualEqual$linear/zero_fraction/total_size/Size%linear/zero_fraction/total_zero/Const*
T0	*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count/SwitchSwitch%linear/zero_fraction/total_zero/Equal%linear/zero_fraction/total_zero/Equal*
T0
*
_output_shapes
: : 
?
3linear/zero_fraction/total_zero/zero_count/switch_tIdentity3linear/zero_fraction/total_zero/zero_count/Switch:1*
T0
*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count/switch_fIdentity1linear/zero_fraction/total_zero/zero_count/Switch*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count/pred_idIdentity%linear/zero_fraction/total_zero/Equal*
T0
*
_output_shapes
: 
?
0linear/zero_fraction/total_zero/zero_count/ConstConst4^linear/zero_fraction/total_zero/zero_count/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpReadVariableOpNlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
?
Nlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/SwitchSwitch1linear/linear_model/age_bucketized/weights/part_02linear/zero_fraction/total_zero/zero_count/pred_id*
T0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
_output_shapes
: : 
?
=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
_output_shapes
: *
dtype0	*
value	B	 R
?
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/yConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
_output_shapes
: *
dtype0	*
valueB	 R????
?
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual	LessEqual=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeDlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/SwitchSwitchBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqualBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_tIdentityFlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_fIdentityDlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_idIdentityBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*
T0*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastCastTlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*
_output_shapes

:
?
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
Ylinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_countSumPlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastQlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/CastCastYlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*
T0*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastCastVlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*
_output_shapes

:
?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/MergeMerge[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countBlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
Olinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/subSub=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeClinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastCastOlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1Cast=linear/zero_fraction/total_zero/zero_count/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truedivRealDivPlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastRlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Alinear/zero_fraction/total_zero/zero_count/zero_fraction/fractionIdentitySlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
/linear/zero_fraction/total_zero/zero_count/CastCast6linear/zero_fraction/total_zero/zero_count/Cast/Switch*

DstT0*

SrcT0	*
_output_shapes
: 
?
6linear/zero_fraction/total_zero/zero_count/Cast/SwitchSwitch$linear/zero_fraction/total_size/Size2linear/zero_fraction/total_zero/zero_count/pred_id*
T0	*7
_class-
+)loc:@linear/zero_fraction/total_size/Size*
_output_shapes
: : 
?
.linear/zero_fraction/total_zero/zero_count/mulMulAlinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction/linear/zero_fraction/total_zero/zero_count/Cast*
T0*
_output_shapes
: 
?
0linear/zero_fraction/total_zero/zero_count/MergeMerge.linear/zero_fraction/total_zero/zero_count/mul0linear/zero_fraction/total_zero/zero_count/Const*
N*
T0*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_1Equal&linear/zero_fraction/total_size/Size_1'linear/zero_fraction/total_zero/Const_1*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_1/SwitchSwitch'linear/zero_fraction/total_zero/Equal_1'linear/zero_fraction/total_zero/Equal_1*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_1/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_1/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_1/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_1/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_1/pred_idIdentity'linear/zero_fraction/total_zero/Equal_1*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_1/ConstConst6^linear/zero_fraction/total_zero/zero_count_1/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
?
Plinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/SwitchSwitchJlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_04linear/zero_fraction/total_zero/zero_count_1/pred_id*
T0*]
_classS
QOloc:@linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_1/switch_f*
_output_shapes
: *
dtype0	*
value	B	 R
?
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_1/switch_f*
_output_shapes
: *
dtype0	*
valueB	 R????
?
Dlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*
_output_shapes

:
?
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*
_output_shapes

:
?
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count_1/CastCast8linear/zero_fraction/total_zero/zero_count_1/Cast/Switch*

DstT0*

SrcT0	*
_output_shapes
: 
?
8linear/zero_fraction/total_zero/zero_count_1/Cast/SwitchSwitch&linear/zero_fraction/total_size/Size_14linear/zero_fraction/total_zero/zero_count_1/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_1*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_1/mulMulClinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fraction1linear/zero_fraction/total_zero/zero_count_1/Cast*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_1/MergeMerge0linear/zero_fraction/total_zero/zero_count_1/mul2linear/zero_fraction/total_zero/zero_count_1/Const*
N*
T0*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_2Equal&linear/zero_fraction/total_size/Size_2'linear/zero_fraction/total_zero/Const_2*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_2/SwitchSwitch'linear/zero_fraction/total_zero/Equal_2'linear/zero_fraction/total_zero/Equal_2*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_2/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_2/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_2/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_2/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_2/pred_idIdentity'linear/zero_fraction/total_zero/Equal_2*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_2/ConstConst6^linear/zero_fraction/total_zero/zero_count_2/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
?
Plinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/education/weights/part_04linear/zero_fraction/total_zero/zero_count_2/pred_id*
T0*?
_class5
31loc:@linear/linear_model/education/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_2/switch_f*
_output_shapes
: *
dtype0	*
value	B	 R
?
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_2/switch_f*
_output_shapes
: *
dtype0	*
valueB	 R????
?
Dlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*
_output_shapes

:
?
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*
_output_shapes

:
?
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count_2/CastCast8linear/zero_fraction/total_zero/zero_count_2/Cast/Switch*

DstT0*

SrcT0	*
_output_shapes
: 
?
8linear/zero_fraction/total_zero/zero_count_2/Cast/SwitchSwitch&linear/zero_fraction/total_size/Size_24linear/zero_fraction/total_zero/zero_count_2/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_2*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_2/mulMulClinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fraction1linear/zero_fraction/total_zero/zero_count_2/Cast*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_2/MergeMerge0linear/zero_fraction/total_zero/zero_count_2/mul2linear/zero_fraction/total_zero/zero_count_2/Const*
N*
T0*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_3Equal&linear/zero_fraction/total_size/Size_3'linear/zero_fraction/total_zero/Const_3*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_3/SwitchSwitch'linear/zero_fraction/total_zero/Equal_3'linear/zero_fraction/total_zero/Equal_3*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_3/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_3/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_3/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_3/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_3/pred_idIdentity'linear/zero_fraction/total_zero/Equal_3*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_3/ConstConst6^linear/zero_fraction/total_zero/zero_count_3/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
?
Plinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/SwitchSwitch9linear/linear_model/education_X_occupation/weights/part_04linear/zero_fraction/total_zero/zero_count_3/pred_id*
T0*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_3/switch_f*
_output_shapes
: *
dtype0	*
value	B	 R
?
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_3/switch_f*
_output_shapes
: *
dtype0	*
valueB	 R????
?
Dlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*
_output_shapes

:
?
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*
_output_shapes

:
?
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count_3/CastCast8linear/zero_fraction/total_zero/zero_count_3/Cast/Switch*

DstT0*

SrcT0	*
_output_shapes
: 
?
8linear/zero_fraction/total_zero/zero_count_3/Cast/SwitchSwitch&linear/zero_fraction/total_size/Size_34linear/zero_fraction/total_zero/zero_count_3/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_3*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_3/mulMulClinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fraction1linear/zero_fraction/total_zero/zero_count_3/Cast*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_3/MergeMerge0linear/zero_fraction/total_zero/zero_count_3/mul2linear/zero_fraction/total_zero/zero_count_3/Const*
N*
T0*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_4Equal&linear/zero_fraction/total_size/Size_4'linear/zero_fraction/total_zero/Const_4*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_4/SwitchSwitch'linear/zero_fraction/total_zero/Equal_4'linear/zero_fraction/total_zero/Equal_4*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_4/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_4/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_4/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_4/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_4/pred_idIdentity'linear/zero_fraction/total_zero/Equal_4*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_4/ConstConst6^linear/zero_fraction/total_zero/zero_count_4/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
?
Plinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/SwitchSwitch1linear/linear_model/marital_status/weights/part_04linear/zero_fraction/total_zero/zero_count_4/pred_id*
T0*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_4/switch_f*
_output_shapes
: *
dtype0	*
value	B	 R
?
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_4/switch_f*
_output_shapes
: *
dtype0	*
valueB	 R????
?
Dlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*
_output_shapes

:
?
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*
_output_shapes

:
?
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count_4/CastCast8linear/zero_fraction/total_zero/zero_count_4/Cast/Switch*

DstT0*

SrcT0	*
_output_shapes
: 
?
8linear/zero_fraction/total_zero/zero_count_4/Cast/SwitchSwitch&linear/zero_fraction/total_size/Size_44linear/zero_fraction/total_zero/zero_count_4/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_4*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_4/mulMulClinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fraction1linear/zero_fraction/total_zero/zero_count_4/Cast*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_4/MergeMerge0linear/zero_fraction/total_zero/zero_count_4/mul2linear/zero_fraction/total_zero/zero_count_4/Const*
N*
T0*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_5Equal&linear/zero_fraction/total_size/Size_5'linear/zero_fraction/total_zero/Const_5*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_5/SwitchSwitch'linear/zero_fraction/total_zero/Equal_5'linear/zero_fraction/total_zero/Equal_5*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_5/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_5/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_5/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_5/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_5/pred_idIdentity'linear/zero_fraction/total_zero/Equal_5*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_5/ConstConst6^linear/zero_fraction/total_zero/zero_count_5/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
?
Plinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/SwitchSwitch-linear/linear_model/occupation/weights/part_04linear/zero_fraction/total_zero/zero_count_5/pred_id*
T0*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_5/switch_f*
_output_shapes
: *
dtype0	*
value	B	 R
?
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_5/switch_f*
_output_shapes
: *
dtype0	*
valueB	 R????
?
Dlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*
_output_shapes

:
?
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*
_output_shapes

:
?
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count_5/CastCast8linear/zero_fraction/total_zero/zero_count_5/Cast/Switch*

DstT0*

SrcT0	*
_output_shapes
: 
?
8linear/zero_fraction/total_zero/zero_count_5/Cast/SwitchSwitch&linear/zero_fraction/total_size/Size_54linear/zero_fraction/total_zero/zero_count_5/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_5*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_5/mulMulClinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fraction1linear/zero_fraction/total_zero/zero_count_5/Cast*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_5/MergeMerge0linear/zero_fraction/total_zero/zero_count_5/mul2linear/zero_fraction/total_zero/zero_count_5/Const*
N*
T0*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_6Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_6Equal&linear/zero_fraction/total_size/Size_6'linear/zero_fraction/total_zero/Const_6*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_6/SwitchSwitch'linear/zero_fraction/total_zero/Equal_6'linear/zero_fraction/total_zero/Equal_6*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_6/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_6/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_6/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_6/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_6/pred_idIdentity'linear/zero_fraction/total_zero/Equal_6*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_6/ConstConst6^linear/zero_fraction/total_zero/zero_count_6/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
?
Plinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/SwitchSwitch/linear/linear_model/relationship/weights/part_04linear/zero_fraction/total_zero/zero_count_6/pred_id*
T0*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_6/switch_f*
_output_shapes
: *
dtype0	*
value	B	 R
?
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_6/switch_f*
_output_shapes
: *
dtype0	*
valueB	 R????
?
Dlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*
_output_shapes

:
?
Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
Xlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*
_output_shapes

:
?
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_6/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count_6/CastCast8linear/zero_fraction/total_zero/zero_count_6/Cast/Switch*

DstT0*

SrcT0	*
_output_shapes
: 
?
8linear/zero_fraction/total_zero/zero_count_6/Cast/SwitchSwitch&linear/zero_fraction/total_size/Size_64linear/zero_fraction/total_zero/zero_count_6/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_6*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_6/mulMulClinear/zero_fraction/total_zero/zero_count_6/zero_fraction/fraction1linear/zero_fraction/total_zero/zero_count_6/Cast*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_6/MergeMerge0linear/zero_fraction/total_zero/zero_count_6/mul2linear/zero_fraction/total_zero/zero_count_6/Const*
N*
T0*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_7Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_7Equal&linear/zero_fraction/total_size/Size_7'linear/zero_fraction/total_zero/Const_7*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_7/SwitchSwitch'linear/zero_fraction/total_zero/Equal_7'linear/zero_fraction/total_zero/Equal_7*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_7/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_7/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_7/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_7/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_7/pred_idIdentity'linear/zero_fraction/total_zero/Equal_7*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_7/ConstConst6^linear/zero_fraction/total_zero/zero_count_7/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/Switch*
_output_shapes

:*
dtype0
?
Plinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/workclass/weights/part_04linear/zero_fraction/total_zero/zero_count_7/pred_id*
T0*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_7/switch_f*
_output_shapes
: *
dtype0	*
value	B	 R
?
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_7/switch_f*
_output_shapes
: *
dtype0	*
valueB	 R????
?
Dlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*
_output_shapes

:
?
Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
Xlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*
_output_shapes

:
?
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_7/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count_7/CastCast8linear/zero_fraction/total_zero/zero_count_7/Cast/Switch*

DstT0*

SrcT0	*
_output_shapes
: 
?
8linear/zero_fraction/total_zero/zero_count_7/Cast/SwitchSwitch&linear/zero_fraction/total_size/Size_74linear/zero_fraction/total_zero/zero_count_7/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_7*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_7/mulMulClinear/zero_fraction/total_zero/zero_count_7/zero_fraction/fraction1linear/zero_fraction/total_zero/zero_count_7/Cast*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_7/MergeMerge0linear/zero_fraction/total_zero/zero_count_7/mul2linear/zero_fraction/total_zero/zero_count_7/Const*
N*
T0*
_output_shapes
: : 
?
$linear/zero_fraction/total_zero/AddNAddN0linear/zero_fraction/total_zero/zero_count/Merge2linear/zero_fraction/total_zero/zero_count_1/Merge2linear/zero_fraction/total_zero/zero_count_2/Merge2linear/zero_fraction/total_zero/zero_count_3/Merge2linear/zero_fraction/total_zero/zero_count_4/Merge2linear/zero_fraction/total_zero/zero_count_5/Merge2linear/zero_fraction/total_zero/zero_count_6/Merge2linear/zero_fraction/total_zero/zero_count_7/Merge*
N*
T0*
_output_shapes
: 
?
)linear/zero_fraction/compute/float32_sizeCast$linear/zero_fraction/total_size/AddN*

DstT0*

SrcT0	*
_output_shapes
: 
?
$linear/zero_fraction/compute/truedivRealDiv$linear/zero_fraction/total_zero/AddN)linear/zero_fraction/compute/float32_size*
T0*
_output_shapes
: 
|
)linear/zero_fraction/zero_fraction_or_nanIdentity$linear/zero_fraction/compute/truediv*
T0*
_output_shapes
: 
?
$linear/fraction_of_zero_weights/tagsConst*
_output_shapes
: *
dtype0*0
value'B% Blinear/fraction_of_zero_weights
?
linear/fraction_of_zero_weightsScalarSummary$linear/fraction_of_zero_weights/tags)linear/zero_fraction/zero_fraction_or_nan*
T0*
_output_shapes
: 
?
linear/zero_fraction_1/SizeSize:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*
_output_shapes
: *
out_type0	
h
"linear/zero_fraction_1/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R????
?
 linear/zero_fraction_1/LessEqual	LessEquallinear/zero_fraction_1/Size"linear/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
?
"linear/zero_fraction_1/cond/SwitchSwitch linear/zero_fraction_1/LessEqual linear/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
w
$linear/zero_fraction_1/cond/switch_tIdentity$linear/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
u
$linear/zero_fraction_1/cond/switch_fIdentity"linear/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
r
#linear/zero_fraction_1/cond/pred_idIdentity linear/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: 
?
/linear/zero_fraction_1/cond/count_nonzero/zerosConst%^linear/zero_fraction_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
?
2linear/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual;linear/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1/linear/zero_fraction_1/cond/count_nonzero/zeros*
T0*'
_output_shapes
:?????????
?
9linear/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitch:linear/linear_model/linear_model/linear_model/weighted_sum#linear/zero_fraction_1/cond/pred_id*
T0*M
_classC
A?loc:@linear/linear_model/linear_model/linear_model/weighted_sum*:
_output_shapes(
&:?????????:?????????
?
.linear/zero_fraction_1/cond/count_nonzero/CastCast2linear/zero_fraction_1/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*'
_output_shapes
:?????????
?
/linear/zero_fraction_1/cond/count_nonzero/ConstConst%^linear/zero_fraction_1/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
?
7linear/zero_fraction_1/cond/count_nonzero/nonzero_countSum.linear/zero_fraction_1/cond/count_nonzero/Cast/linear/zero_fraction_1/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
 linear/zero_fraction_1/cond/CastCast7linear/zero_fraction_1/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 
?
1linear/zero_fraction_1/cond/count_nonzero_1/zerosConst%^linear/zero_fraction_1/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
?
4linear/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual;linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch1linear/zero_fraction_1/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:?????????
?
;linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitch:linear/linear_model/linear_model/linear_model/weighted_sum#linear/zero_fraction_1/cond/pred_id*
T0*M
_classC
A?loc:@linear/linear_model/linear_model/linear_model/weighted_sum*:
_output_shapes(
&:?????????:?????????
?
0linear/zero_fraction_1/cond/count_nonzero_1/CastCast4linear/zero_fraction_1/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*'
_output_shapes
:?????????
?
1linear/zero_fraction_1/cond/count_nonzero_1/ConstConst%^linear/zero_fraction_1/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
?
9linear/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum0linear/zero_fraction_1/cond/count_nonzero_1/Cast1linear/zero_fraction_1/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
!linear/zero_fraction_1/cond/MergeMerge9linear/zero_fraction_1/cond/count_nonzero_1/nonzero_count linear/zero_fraction_1/cond/Cast*
N*
T0	*
_output_shapes
: : 
?
-linear/zero_fraction_1/counts_to_fraction/subSublinear/zero_fraction_1/Size!linear/zero_fraction_1/cond/Merge*
T0	*
_output_shapes
: 
?
.linear/zero_fraction_1/counts_to_fraction/CastCast-linear/zero_fraction_1/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
?
0linear/zero_fraction_1/counts_to_fraction/Cast_1Castlinear/zero_fraction_1/Size*

DstT0*

SrcT0	*
_output_shapes
: 
?
1linear/zero_fraction_1/counts_to_fraction/truedivRealDiv.linear/zero_fraction_1/counts_to_fraction/Cast0linear/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 

linear/zero_fraction_1/fractionIdentity1linear/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
*linear/linear/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*6
value-B+ B%linear/linear/fraction_of_zero_values
?
%linear/linear/fraction_of_zero_valuesScalarSummary*linear/linear/fraction_of_zero_values/tagslinear/zero_fraction_1/fraction*
T0*
_output_shapes
: 
u
linear/linear/activation/tagConst*
_output_shapes
: *
dtype0*)
value B Blinear/linear/activation
?
linear/linear/activationHistogramSummarylinear/linear/activation/tag:linear/linear_model/linear_model/linear_model/weighted_sum*
_output_shapes
: 
?
addAddV2dnn/logits/BiasAdd:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*'
_output_shapes
:?????????
P
head/predictions/logits/ShapeShapeadd*
T0*
_output_shapes
:
s
1head/predictions/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :
c
[head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
T
Lhead/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
[
head/predictions/logisticSigmoidadd*
T0*'
_output_shapes
:?????????
_
head/predictions/zeros_like	ZerosLikeadd*
T0*'
_output_shapes
:?????????
q
&head/predictions/two_class_logits/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
!head/predictions/two_class_logitsConcatV2head/predictions/zeros_likeadd&head/predictions/two_class_logits/axis*
N*
T0*'
_output_shapes
:?????????
~
head/predictions/probabilitiesSoftmax!head/predictions/two_class_logits*
T0*'
_output_shapes
:?????????
o
$head/predictions/class_ids/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
head/predictions/class_idsArgMax!head/predictions/two_class_logits$head/predictions/class_ids/dimension*
T0*#
_output_shapes
:?????????
j
head/predictions/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:?????????
I
head/predictions/ShapeShapeadd*
T0*
_output_shapes
:
n
$head/predictions/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
p
&head/predictions/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
p
&head/predictions/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
head/predictions/strided_sliceStridedSlicehead/predictions/Shape$head/predictions/strided_slice/stack&head/predictions/strided_slice/stack_1&head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
^
head/predictions/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
^
head/predictions/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
^
head/predictions/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/rangeRangehead/predictions/range/starthead/predictions/range/limithead/predictions/range/delta*
_output_shapes
:
c
!head/predictions/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
head/predictions/ExpandDims_1
ExpandDimshead/predictions/range!head/predictions/ExpandDims_1/dim*
T0*
_output_shapes

:
c
!head/predictions/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/Tile/multiplesPackhead/predictions/strided_slice!head/predictions/Tile/multiples/1*
N*
T0*
_output_shapes
:
?
head/predictions/TileTilehead/predictions/ExpandDims_1head/predictions/Tile/multiples*
T0*'
_output_shapes
:?????????
K
head/predictions/Shape_1Shapeadd*
T0*
_output_shapes
:
p
&head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
r
(head/predictions/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
r
(head/predictions/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
 head/predictions/strided_slice_1StridedSlicehead/predictions/Shape_1&head/predictions/strided_slice_1/stack(head/predictions/strided_slice_1/stack_1(head/predictions/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
`
head/predictions/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 
`
head/predictions/range_1/limitConst*
_output_shapes
: *
dtype0*
value	B :
`
head/predictions/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :
?
head/predictions/range_1Rangehead/predictions/range_1/starthead/predictions/range_1/limithead/predictions/range_1/delta*
_output_shapes
:
d
head/predictions/AsStringAsStringhead/predictions/range_1*
T0*
_output_shapes
:
c
!head/predictions/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
head/predictions/ExpandDims_2
ExpandDimshead/predictions/AsString!head/predictions/ExpandDims_2/dim*
T0*
_output_shapes

:
e
#head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
?
!head/predictions/Tile_1/multiplesPack head/predictions/strided_slice_1#head/predictions/Tile_1/multiples/1*
N*
T0*
_output_shapes
:
?
head/predictions/Tile_1Tilehead/predictions/ExpandDims_2!head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:?????????
w
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*'
_output_shapes
:?????????
X

head/ShapeShapehead/predictions/probabilities*
T0*
_output_shapes
:
b
head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
d
head/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
d
head/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
R
head/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
R
head/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
R
head/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
e

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:
J
head/AsStringAsString
head/range*
T0*
_output_shapes
:
U
head/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
j
head/ExpandDims
ExpandDimshead/AsStringhead/ExpandDims/dim*
T0*
_output_shapes

:
W
head/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
t
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
N*
T0*
_output_shapes
:
i
	head/TileTilehead/ExpandDimshead/Tile/multiples*
T0*'
_output_shapes
:?????????

initNoOp
?
init_all_tablesNoOpz^dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/table_init/LookupTableImportV2?^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/table_init/LookupTableImportV2?^dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/table_init/LookupTableImportV2s^linear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/table_init/LookupTableImportV2}^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/table_init/LookupTableImportV2y^linear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/table_init/LookupTableImportV2

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
f
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
f
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
_output_shapes	
:?*
dtype0
h
save/IdentityIdentitysave/Read/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes	
:?
_
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes	
:?
?
save/Read_1/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
_output_shapes
:	2?*
dtype0
p
save/Identity_2Identitysave/Read_1/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	2?
e
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes
:	2?
?
save/Read_2/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0"/device:CPU:0*
_output_shapes
:@*
dtype0
k
save/Identity_4Identitysave/Read_2/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:@
`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes
:@
?
save/Read_3/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0"/device:CPU:0*
_output_shapes
:	?@*
dtype0
p
save/Identity_6Identitysave/Read_3/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	?@
e
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
T0*
_output_shapes
:	?@
?
save/Read_4/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
k
save/Identity_8Identitysave/Read_4/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
T0*
_output_shapes
:
?
save/Read_5/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0"/device:CPU:0*
_output_shapes

:@*
dtype0
p
save/Identity_10Identitysave/Read_5/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:@
f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:@
|
save/Read_6/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
l
save/Identity_12Identitysave/Read_6/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes
:
?
save/Read_7/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
p
save/Identity_14Identitysave/Read_7/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_8/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
p
save/Identity_16Identitysave/Read_8/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_9/ReadVariableOpReadVariableOpJlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
p
save/Identity_18Identitysave/Read_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_10/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
m
save/Identity_20Identitysave/Read_10/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
T0*
_output_shapes
:
?
save/Read_11/ReadVariableOpReadVariableOp,linear/linear_model/education/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
q
save/Identity_22Identitysave/Read_11/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_12/ReadVariableOpReadVariableOp9linear/linear_model/education_X_occupation/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
q
save/Identity_24Identitysave/Read_12/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_25Identitysave/Identity_24"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_13/ReadVariableOpReadVariableOp1linear/linear_model/marital_status/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
q
save/Identity_26Identitysave/Read_13/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_27Identitysave/Identity_26"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_14/ReadVariableOpReadVariableOp-linear/linear_model/occupation/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
q
save/Identity_28Identitysave/Read_14/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_29Identitysave/Identity_28"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_15/ReadVariableOpReadVariableOp/linear/linear_model/relationship/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
q
save/Identity_30Identitysave/Read_15/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_31Identitysave/Identity_30"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_16/ReadVariableOpReadVariableOp,linear/linear_model/workclass/weights/part_0"/device:CPU:0*
_output_shapes

:*
dtype0
q
save/Identity_32Identitysave/Read_16/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_33Identitysave/Identity_32"/device:CPU:0*
T0*
_output_shapes

:
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBQdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weightsBVdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_mapsBPdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weightsBUdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_mapsBdnn/logits/biasBdnn/logits/kernelBglobal_stepB*linear/linear_model/age_bucketized/weightsBClinear/linear_model/age_bucketized_X_education_X_occupation/weightsB linear/linear_model/bias_weightsB%linear/linear_model/education/weightsB2linear/linear_model/education_X_occupation/weightsB*linear/linear_model/marital_status/weightsB&linear/linear_model/occupation/weightsB(linear/linear_model/relationship/weightsB%linear/linear_model/workclass/weights
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B	128 0,128B50 128 0,50:0,128B64 0,64B128 64 0,128:0,64B25 0,25B64 25 0,64:0,25B6 8 0,6:0,8B13 0,13B3 8 0,3:0,8B13 0,13B1 0,1B25 1 0,25:0,1B B11 1 0,11:0,1B13 1 0,13:0,1B1 0,1B16 1 0,16:0,1B13 1 0,13:0,1B7 1 0,7:0,1B13 1 0,13:0,1B6 1 0,6:0,1B13 1 0,13:0,1
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicessave/Identity_1save/Identity_3save/Identity_5save/Identity_7save/Identity_9save/Identity_11]dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/readbdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/read\dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/readadnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/readsave/Identity_13save/Identity_15global_stepsave/Identity_17save/Identity_19save/Identity_21save/Identity_23save/Identity_25save/Identity_27save/Identity_29save/Identity_31save/Identity_33"/device:CPU:0*$
dtypes
2			
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/Identity_34Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBQdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weightsBVdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_mapsBPdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weightsBUdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_mapsBdnn/logits/biasBdnn/logits/kernelBglobal_stepB*linear/linear_model/age_bucketized/weightsBClinear/linear_model/age_bucketized_X_education_X_occupation/weightsB linear/linear_model/bias_weightsB%linear/linear_model/education/weightsB2linear/linear_model/education_X_occupation/weightsB*linear/linear_model/marital_status/weightsB&linear/linear_model/occupation/weightsB(linear/linear_model/relationship/weightsB%linear/linear_model/workclass/weights
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B	128 0,128B50 128 0,50:0,128B64 0,64B128 64 0,128:0,64B25 0,25B64 25 0,64:0,25B6 8 0,6:0,8B13 0,13B3 8 0,3:0,8B13 0,13B1 0,1B25 1 0,25:0,1B B11 1 0,11:0,1B13 1 0,13:0,1B1 0,1B16 1 0,16:0,1B13 1 0,13:0,1B7 1 0,7:0,1B13 1 0,13:0,1B6 1 0,6:0,1B13 1 0,13:0,1
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?:?:	2?:@:	?@::@::::::::::::::::*$
dtypes
2			
R
save/Identity_35Identitysave/RestoreV2*
T0*
_output_shapes	
:?
g
save/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_35*
dtype0
X
save/Identity_36Identitysave/RestoreV2:1*
T0*
_output_shapes
:	2?
k
save/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_36*
dtype0
S
save/Identity_37Identitysave/RestoreV2:2*
T0*
_output_shapes
:@
i
save/AssignVariableOp_2AssignVariableOpdnn/hiddenlayer_1/bias/part_0save/Identity_37*
dtype0
X
save/Identity_38Identitysave/RestoreV2:3*
T0*
_output_shapes
:	?@
k
save/AssignVariableOp_3AssignVariableOpdnn/hiddenlayer_1/kernel/part_0save/Identity_38*
dtype0
S
save/Identity_39Identitysave/RestoreV2:4*
T0*
_output_shapes
:
i
save/AssignVariableOp_4AssignVariableOpdnn/hiddenlayer_2/bias/part_0save/Identity_39*
dtype0
W
save/Identity_40Identitysave/RestoreV2:5*
T0*
_output_shapes

:@
k
save/AssignVariableOp_5AssignVariableOpdnn/hiddenlayer_2/kernel/part_0save/Identity_40*
dtype0
?
save/AssignAssignXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0save/RestoreV2:6*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:
?
save/Assign_1Assign]dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0save/RestoreV2:7*
T0	*p
_classf
dbloc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0*
_output_shapes
:
?
save/Assign_2AssignWdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0save/RestoreV2:8*
T0*j
_class`
^\loc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0*
_output_shapes

:
?
save/Assign_3Assign\dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0save/RestoreV2:9*
T0	*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0*
_output_shapes
:
T
save/Identity_41Identitysave/RestoreV2:10*
T0*
_output_shapes
:
b
save/AssignVariableOp_6AssignVariableOpdnn/logits/bias/part_0save/Identity_41*
dtype0
X
save/Identity_42Identitysave/RestoreV2:11*
T0*
_output_shapes

:
d
save/AssignVariableOp_7AssignVariableOpdnn/logits/kernel/part_0save/Identity_42*
dtype0
x
save/Assign_4Assignglobal_stepsave/RestoreV2:12*
T0	*
_class
loc:@global_step*
_output_shapes
: 
X
save/Identity_43Identitysave/RestoreV2:13*
T0*
_output_shapes

:
}
save/AssignVariableOp_8AssignVariableOp1linear/linear_model/age_bucketized/weights/part_0save/Identity_43*
dtype0
X
save/Identity_44Identitysave/RestoreV2:14*
T0*
_output_shapes

:
?
save/AssignVariableOp_9AssignVariableOpJlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0save/Identity_44*
dtype0
T
save/Identity_45Identitysave/RestoreV2:15*
T0*
_output_shapes
:
t
save/AssignVariableOp_10AssignVariableOp'linear/linear_model/bias_weights/part_0save/Identity_45*
dtype0
X
save/Identity_46Identitysave/RestoreV2:16*
T0*
_output_shapes

:
y
save/AssignVariableOp_11AssignVariableOp,linear/linear_model/education/weights/part_0save/Identity_46*
dtype0
X
save/Identity_47Identitysave/RestoreV2:17*
T0*
_output_shapes

:
?
save/AssignVariableOp_12AssignVariableOp9linear/linear_model/education_X_occupation/weights/part_0save/Identity_47*
dtype0
X
save/Identity_48Identitysave/RestoreV2:18*
T0*
_output_shapes

:
~
save/AssignVariableOp_13AssignVariableOp1linear/linear_model/marital_status/weights/part_0save/Identity_48*
dtype0
X
save/Identity_49Identitysave/RestoreV2:19*
T0*
_output_shapes

:
z
save/AssignVariableOp_14AssignVariableOp-linear/linear_model/occupation/weights/part_0save/Identity_49*
dtype0
X
save/Identity_50Identitysave/RestoreV2:20*
T0*
_output_shapes

:
|
save/AssignVariableOp_15AssignVariableOp/linear/linear_model/relationship/weights/part_0save/Identity_50*
dtype0
X
save/Identity_51Identitysave/RestoreV2:21*
T0*
_output_shapes

:
y
save/AssignVariableOp_16AssignVariableOp,linear/linear_model/workclass/weights/part_0save/Identity_51*
dtype0
?
save/restore_shardNoOp^save/Assign^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4
-
save/restore_allNoOp^save/restore_shard"??
save/Const:0save/Identity_34:0save/restore_all (5 @F8"??
cond_context????
?
 dnn/zero_fraction/cond/cond_text dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_t:0 *?
dnn/hiddenlayer_0/Relu:0
dnn/zero_fraction/cond/Cast:0
+dnn/zero_fraction/cond/count_nonzero/Cast:0
,dnn/zero_fraction/cond/count_nonzero/Const:0
6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
/dnn/zero_fraction/cond/count_nonzero/NotEqual:0
4dnn/zero_fraction/cond/count_nonzero/nonzero_count:0
,dnn/zero_fraction/cond/count_nonzero/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_t:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0R
dnn/hiddenlayer_0/Relu:06dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
?
"dnn/zero_fraction/cond/cond_text_1 dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_f:0*?
dnn/hiddenlayer_0/Relu:0
-dnn/zero_fraction/cond/count_nonzero_1/Cast:0
.dnn/zero_fraction/cond/count_nonzero_1/Const:0
8dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
1dnn/zero_fraction/cond/count_nonzero_1/NotEqual:0
6dnn/zero_fraction/cond/count_nonzero_1/nonzero_count:0
.dnn/zero_fraction/cond/count_nonzero_1/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_f:0T
dnn/hiddenlayer_0/Relu:08dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0
?
"dnn/zero_fraction_1/cond/cond_text"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_t:0 *?
dnn/hiddenlayer_1/Relu:0
dnn/zero_fraction_1/cond/Cast:0
-dnn/zero_fraction_1/cond/count_nonzero/Cast:0
.dnn/zero_fraction_1/cond/count_nonzero/Const:0
8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_1/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_1/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_1/cond/count_nonzero/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_t:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0T
dnn/hiddenlayer_1/Relu:08dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_1/cond/cond_text_1"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_f:0*?
dnn/hiddenlayer_1/Relu:0
/dnn/zero_fraction_1/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_1/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_1/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_f:0V
dnn/hiddenlayer_1/Relu:0:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0
?
"dnn/zero_fraction_2/cond/cond_text"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_t:0 *?
dnn/hiddenlayer_2/Relu:0
dnn/zero_fraction_2/cond/Cast:0
-dnn/zero_fraction_2/cond/count_nonzero/Cast:0
.dnn/zero_fraction_2/cond/count_nonzero/Const:0
8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_2/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_2/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_2/cond/count_nonzero/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_t:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0T
dnn/hiddenlayer_2/Relu:08dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_2/cond/cond_text_1"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_f:0*?
dnn/hiddenlayer_2/Relu:0
/dnn/zero_fraction_2/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_2/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_2/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_f:0V
dnn/hiddenlayer_2/Relu:0:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0
?
"dnn/zero_fraction_3/cond/cond_text"dnn/zero_fraction_3/cond/pred_id:0#dnn/zero_fraction_3/cond/switch_t:0 *?
dnn/logits/BiasAdd:0
dnn/zero_fraction_3/cond/Cast:0
-dnn/zero_fraction_3/cond/count_nonzero/Cast:0
.dnn/zero_fraction_3/cond/count_nonzero/Const:0
8dnn/zero_fraction_3/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_3/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_3/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_3/cond/count_nonzero/zeros:0
"dnn/zero_fraction_3/cond/pred_id:0
#dnn/zero_fraction_3/cond/switch_t:0H
"dnn/zero_fraction_3/cond/pred_id:0"dnn/zero_fraction_3/cond/pred_id:0P
dnn/logits/BiasAdd:08dnn/zero_fraction_3/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_3/cond/cond_text_1"dnn/zero_fraction_3/cond/pred_id:0#dnn/zero_fraction_3/cond/switch_f:0*?
dnn/logits/BiasAdd:0
/dnn/zero_fraction_3/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_3/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_3/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_3/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_3/cond/pred_id:0
#dnn/zero_fraction_3/cond/switch_f:0H
"dnn/zero_fraction_3/cond/pred_id:0"dnn/zero_fraction_3/cond/pred_id:0R
dnn/logits/BiasAdd:0:dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/Switch:0
?
4linear/zero_fraction/total_zero/zero_count/cond_text4linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_t:0 *?
2linear/zero_fraction/total_zero/zero_count/Const:0
4linear/zero_fraction/total_zero/zero_count/pred_id:0
5linear/zero_fraction/total_zero/zero_count/switch_t:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:0
?.
6linear/zero_fraction/total_zero/zero_count/cond_text_14linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_f:0*?
3linear/linear_model/age_bucketized/weights/part_0:0
&linear/zero_fraction/total_size/Size:0
8linear/zero_fraction/total_zero/zero_count/Cast/Switch:0
1linear/zero_fraction/total_zero/zero_count/Cast:0
0linear/zero_fraction/total_zero/zero_count/mul:0
4linear/zero_fraction/total_zero/zero_count/pred_id:0
5linear/zero_fraction/total_zero/zero_count/switch_f:0
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/y:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual:0
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch:0
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
?linear/zero_fraction/total_zero/zero_count/zero_fraction/Size:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast:0
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge:0
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge:1
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:0
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:1
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Cast:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual:0
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Cast:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const:0
_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Xlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1:0
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/sub:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truediv:0
Clinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction:0?
3linear/linear_model/age_bucketized/weights/part_0:0Plinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch:0b
&linear/zero_fraction/total_size/Size:08linear/zero_fraction/total_zero/zero_count/Cast/Switch:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:02?

?

Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_textGlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0 *?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast:0
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Cast:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual:0
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:12?

?

Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_text_1Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0*?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Cast:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const:0
_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Xlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
?
6linear/zero_fraction/total_zero/zero_count_1/cond_text6linear/zero_fraction/total_zero/zero_count_1/pred_id:07linear/zero_fraction/total_zero/zero_count_1/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_1/Const:0
6linear/zero_fraction/total_zero/zero_count_1/pred_id:0
7linear/zero_fraction/total_zero/zero_count_1/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_1/pred_id:06linear/zero_fraction/total_zero/zero_count_1/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_1/cond_text_16linear/zero_fraction/total_zero/zero_count_1/pred_id:07linear/zero_fraction/total_zero/zero_count_1/switch_f:0*?
Llinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0:0
(linear/zero_fraction/total_size/Size_1:0
:linear/zero_fraction/total_zero/zero_count_1/Cast/Switch:0
3linear/zero_fraction/total_zero/zero_count_1/Cast:0
2linear/zero_fraction/total_zero/zero_count_1/mul:0
6linear/zero_fraction/total_zero/zero_count_1/pred_id:0
7linear/zero_fraction/total_zero/zero_count_1/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fraction:0?
Llinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch:0p
6linear/zero_fraction/total_zero/zero_count_1/pred_id:06linear/zero_fraction/total_zero/zero_count_1/pred_id:0f
(linear/zero_fraction/total_size/Size_1:0:linear/zero_fraction/total_zero/zero_count_1/Cast/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_2/cond_text6linear/zero_fraction/total_zero/zero_count_2/pred_id:07linear/zero_fraction/total_zero/zero_count_2/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_2/Const:0
6linear/zero_fraction/total_zero/zero_count_2/pred_id:0
7linear/zero_fraction/total_zero/zero_count_2/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_2/pred_id:06linear/zero_fraction/total_zero/zero_count_2/pred_id:0
?/
8linear/zero_fraction/total_zero/zero_count_2/cond_text_16linear/zero_fraction/total_zero/zero_count_2/pred_id:07linear/zero_fraction/total_zero/zero_count_2/switch_f:0*?
.linear/linear_model/education/weights/part_0:0
(linear/zero_fraction/total_size/Size_2:0
:linear/zero_fraction/total_zero/zero_count_2/Cast/Switch:0
3linear/zero_fraction/total_zero/zero_count_2/Cast:0
2linear/zero_fraction/total_zero/zero_count_2/mul:0
6linear/zero_fraction/total_zero/zero_count_2/pred_id:0
7linear/zero_fraction/total_zero/zero_count_2/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fraction:0?
.linear/linear_model/education/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch:0f
(linear/zero_fraction/total_size/Size_2:0:linear/zero_fraction/total_zero/zero_count_2/Cast/Switch:0p
6linear/zero_fraction/total_zero/zero_count_2/pred_id:06linear/zero_fraction/total_zero/zero_count_2/pred_id:02?

?

Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_3/cond_text6linear/zero_fraction/total_zero/zero_count_3/pred_id:07linear/zero_fraction/total_zero/zero_count_3/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_3/Const:0
6linear/zero_fraction/total_zero/zero_count_3/pred_id:0
7linear/zero_fraction/total_zero/zero_count_3/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_3/pred_id:06linear/zero_fraction/total_zero/zero_count_3/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_3/cond_text_16linear/zero_fraction/total_zero/zero_count_3/pred_id:07linear/zero_fraction/total_zero/zero_count_3/switch_f:0*?
;linear/linear_model/education_X_occupation/weights/part_0:0
(linear/zero_fraction/total_size/Size_3:0
:linear/zero_fraction/total_zero/zero_count_3/Cast/Switch:0
3linear/zero_fraction/total_zero/zero_count_3/Cast:0
2linear/zero_fraction/total_zero/zero_count_3/mul:0
6linear/zero_fraction/total_zero/zero_count_3/pred_id:0
7linear/zero_fraction/total_zero/zero_count_3/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_3/pred_id:06linear/zero_fraction/total_zero/zero_count_3/pred_id:0?
;linear/linear_model/education_X_occupation/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch:0f
(linear/zero_fraction/total_size/Size_3:0:linear/zero_fraction/total_zero/zero_count_3/Cast/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0?
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0?
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:12?

?

Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_4/cond_text6linear/zero_fraction/total_zero/zero_count_4/pred_id:07linear/zero_fraction/total_zero/zero_count_4/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_4/Const:0
6linear/zero_fraction/total_zero/zero_count_4/pred_id:0
7linear/zero_fraction/total_zero/zero_count_4/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_4/pred_id:06linear/zero_fraction/total_zero/zero_count_4/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_4/cond_text_16linear/zero_fraction/total_zero/zero_count_4/pred_id:07linear/zero_fraction/total_zero/zero_count_4/switch_f:0*?
3linear/linear_model/marital_status/weights/part_0:0
(linear/zero_fraction/total_size/Size_4:0
:linear/zero_fraction/total_zero/zero_count_4/Cast/Switch:0
3linear/zero_fraction/total_zero/zero_count_4/Cast:0
2linear/zero_fraction/total_zero/zero_count_4/mul:0
6linear/zero_fraction/total_zero/zero_count_4/pred_id:0
7linear/zero_fraction/total_zero/zero_count_4/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_4/pred_id:06linear/zero_fraction/total_zero/zero_count_4/pred_id:0f
(linear/zero_fraction/total_size/Size_4:0:linear/zero_fraction/total_zero/zero_count_4/Cast/Switch:0?
3linear/linear_model/marital_status/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0?
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0?
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:12?

?

Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_5/cond_text6linear/zero_fraction/total_zero/zero_count_5/pred_id:07linear/zero_fraction/total_zero/zero_count_5/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_5/Const:0
6linear/zero_fraction/total_zero/zero_count_5/pred_id:0
7linear/zero_fraction/total_zero/zero_count_5/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_5/pred_id:06linear/zero_fraction/total_zero/zero_count_5/pred_id:0
?/
8linear/zero_fraction/total_zero/zero_count_5/cond_text_16linear/zero_fraction/total_zero/zero_count_5/pred_id:07linear/zero_fraction/total_zero/zero_count_5/switch_f:0*?
/linear/linear_model/occupation/weights/part_0:0
(linear/zero_fraction/total_size/Size_5:0
:linear/zero_fraction/total_zero/zero_count_5/Cast/Switch:0
3linear/zero_fraction/total_zero/zero_count_5/Cast:0
2linear/zero_fraction/total_zero/zero_count_5/mul:0
6linear/zero_fraction/total_zero/zero_count_5/pred_id:0
7linear/zero_fraction/total_zero/zero_count_5/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fraction:0f
(linear/zero_fraction/total_size/Size_5:0:linear/zero_fraction/total_zero/zero_count_5/Cast/Switch:0p
6linear/zero_fraction/total_zero/zero_count_5/pred_id:06linear/zero_fraction/total_zero/zero_count_5/pred_id:0?
/linear/linear_model/occupation/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0?
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0?
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:12?

?

Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0?
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0?
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
?
6linear/zero_fraction/total_zero/zero_count_6/cond_text6linear/zero_fraction/total_zero/zero_count_6/pred_id:07linear/zero_fraction/total_zero/zero_count_6/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_6/Const:0
6linear/zero_fraction/total_zero/zero_count_6/pred_id:0
7linear/zero_fraction/total_zero/zero_count_6/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_6/pred_id:06linear/zero_fraction/total_zero/zero_count_6/pred_id:0
?/
8linear/zero_fraction/total_zero/zero_count_6/cond_text_16linear/zero_fraction/total_zero/zero_count_6/pred_id:07linear/zero_fraction/total_zero/zero_count_6/switch_f:0*?
1linear/linear_model/relationship/weights/part_0:0
(linear/zero_fraction/total_size/Size_6:0
:linear/zero_fraction/total_zero/zero_count_6/Cast/Switch:0
3linear/zero_fraction/total_zero/zero_count_6/Cast:0
2linear/zero_fraction/total_zero/zero_count_6/mul:0
6linear/zero_fraction/total_zero/zero_count_6/pred_id:0
7linear/zero_fraction/total_zero/zero_count_6/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_6/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_6/pred_id:06linear/zero_fraction/total_zero/zero_count_6/pred_id:0f
(linear/zero_fraction/total_size/Size_6:0:linear/zero_fraction/total_zero/zero_count_6/Cast/Switch:0?
1linear/linear_model/relationship/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t:0?
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0?
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:12?

?

Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f:0?
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0?
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
?
6linear/zero_fraction/total_zero/zero_count_7/cond_text6linear/zero_fraction/total_zero/zero_count_7/pred_id:07linear/zero_fraction/total_zero/zero_count_7/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_7/Const:0
6linear/zero_fraction/total_zero/zero_count_7/pred_id:0
7linear/zero_fraction/total_zero/zero_count_7/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_7/pred_id:06linear/zero_fraction/total_zero/zero_count_7/pred_id:0
?/
8linear/zero_fraction/total_zero/zero_count_7/cond_text_16linear/zero_fraction/total_zero/zero_count_7/pred_id:07linear/zero_fraction/total_zero/zero_count_7/switch_f:0*?
.linear/linear_model/workclass/weights/part_0:0
(linear/zero_fraction/total_size/Size_7:0
:linear/zero_fraction/total_zero/zero_count_7/Cast/Switch:0
3linear/zero_fraction/total_zero/zero_count_7/Cast:0
2linear/zero_fraction/total_zero/zero_count_7/mul:0
6linear/zero_fraction/total_zero/zero_count_7/pred_id:0
7linear/zero_fraction/total_zero/zero_count_7/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_7/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_7/pred_id:06linear/zero_fraction/total_zero/zero_count_7/pred_id:0f
(linear/zero_fraction/total_size/Size_7:0:linear/zero_fraction/total_zero/zero_count_7/Cast/Switch:0?
.linear/linear_model/workclass/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
?
%linear/zero_fraction_1/cond/cond_text%linear/zero_fraction_1/cond/pred_id:0&linear/zero_fraction_1/cond/switch_t:0 *?
<linear/linear_model/linear_model/linear_model/weighted_sum:0
"linear/zero_fraction_1/cond/Cast:0
0linear/zero_fraction_1/cond/count_nonzero/Cast:0
1linear/zero_fraction_1/cond/count_nonzero/Const:0
;linear/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
4linear/zero_fraction_1/cond/count_nonzero/NotEqual:0
9linear/zero_fraction_1/cond/count_nonzero/nonzero_count:0
1linear/zero_fraction_1/cond/count_nonzero/zeros:0
%linear/zero_fraction_1/cond/pred_id:0
&linear/zero_fraction_1/cond/switch_t:0{
<linear/linear_model/linear_model/linear_model/weighted_sum:0;linear/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1N
%linear/zero_fraction_1/cond/pred_id:0%linear/zero_fraction_1/cond/pred_id:0
?
'linear/zero_fraction_1/cond/cond_text_1%linear/zero_fraction_1/cond/pred_id:0&linear/zero_fraction_1/cond/switch_f:0*?
<linear/linear_model/linear_model/linear_model/weighted_sum:0
2linear/zero_fraction_1/cond/count_nonzero_1/Cast:0
3linear/zero_fraction_1/cond/count_nonzero_1/Const:0
=linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
6linear/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
;linear/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
3linear/zero_fraction_1/cond/count_nonzero_1/zeros:0
%linear/zero_fraction_1/cond/pred_id:0
&linear/zero_fraction_1/cond/switch_f:0}
<linear/linear_model/linear_model/linear_model/weighted_sum:0=linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0N
%linear/zero_fraction_1/cond/pred_id:0%linear/zero_fraction_1/cond/pred_id:0"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"%
saved_model_main_op


group_deps"?
	summaries?
?
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
/dnn/dnn/hiddenlayer_2/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_2/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0
linear/bias:0
!linear/fraction_of_zero_weights:0
'linear/linear/fraction_of_zero_values:0
linear/linear/activation:0"?
table_initializer?
?
ydnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/table_init/LookupTableImportV2
?dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/table_init/LookupTableImportV2
dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/table_init/LookupTableImportV2
rlinear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/table_init/LookupTableImportV2
|linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/table_init/LookupTableImportV2
xlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/table_init/LookupTableImportV2"?4
trainable_variables?4?4
?
Zdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights  "2udnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform:08
?
_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0:0ddnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/Assignddnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/read:0"a
Vdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps "2qdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/Initializer/zeros:08
?
Ydnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights  "2kdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/Initializer/zeros:08
?
^dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0:0cdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/Assigncdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/read:0"`
Udnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps "2pdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"(
dnn/hiddenlayer_0/kernel2?  "2?(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"#
dnn/hiddenlayer_0/bias? "?(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"(
dnn/hiddenlayer_1/kernel?@  "?@(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias@ "@(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign5dnn/hiddenlayer_2/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_2/kernel@  "@(2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign3dnn/hiddenlayer_2/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_2/bias "(21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:08
?
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel  "(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
?
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08
?
3linear/linear_model/age_bucketized/weights/part_0:08linear/linear_model/age_bucketized/weights/part_0/AssignGlinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/age_bucketized/weights  "(2Elinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros:08
?
Llinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0:0Qlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/Assign`linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/Read/ReadVariableOp:0"Q
Clinear/linear_model/age_bucketized_X_education_X_occupation/weights  "(2^linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/education/weights/part_0:03linear/linear_model/education/weights/part_0/AssignBlinear/linear_model/education/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/education/weights  "(2@linear/linear_model/education/weights/part_0/Initializer/zeros:08
?
;linear/linear_model/education_X_occupation/weights/part_0:0@linear/linear_model/education_X_occupation/weights/part_0/AssignOlinear/linear_model/education_X_occupation/weights/part_0/Read/ReadVariableOp:0"@
2linear/linear_model/education_X_occupation/weights  "(2Mlinear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros:08
?
3linear/linear_model/marital_status/weights/part_0:08linear/linear_model/marital_status/weights/part_0/AssignGlinear/linear_model/marital_status/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/marital_status/weights  "(2Elinear/linear_model/marital_status/weights/part_0/Initializer/zeros:08
?
/linear/linear_model/occupation/weights/part_0:04linear/linear_model/occupation/weights/part_0/AssignClinear/linear_model/occupation/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/occupation/weights  "(2Alinear/linear_model/occupation/weights/part_0/Initializer/zeros:08
?
1linear/linear_model/relationship/weights/part_0:06linear/linear_model/relationship/weights/part_0/AssignElinear/linear_model/relationship/weights/part_0/Read/ReadVariableOp:0"6
(linear/linear_model/relationship/weights  "(2Clinear/linear_model/relationship/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/workclass/weights/part_0:03linear/linear_model/workclass/weights/part_0/AssignBlinear/linear_model/workclass/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/workclass/weights  "(2@linear/linear_model/workclass/weights/part_0/Initializer/zeros:08
?
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"?5
	variables?5?5
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
?
Zdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Assign_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/read:0"_
Qdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights  "2udnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/random_uniform:08
?
_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0:0ddnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/Assignddnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/read:0"a
Vdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps "2qdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights_maps/part_0/Initializer/zeros:08
?
Ydnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0:0^dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/Assign^dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/read:0"^
Pdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights  "2kdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights/part_0/Initializer/zeros:08
?
^dnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0:0cdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/Assigncdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/read:0"`
Udnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps "2pdnn/input_from_feature_columns/input_layer/workclass_embedding/embedding_weights_maps/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"(
dnn/hiddenlayer_0/kernel2?  "2?(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"#
dnn/hiddenlayer_0/bias? "?(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"(
dnn/hiddenlayer_1/kernel?@  "?@(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias@ "@(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign5dnn/hiddenlayer_2/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_2/kernel@  "@(2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign3dnn/hiddenlayer_2/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_2/bias "(21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:08
?
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel  "(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
?
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08
?
3linear/linear_model/age_bucketized/weights/part_0:08linear/linear_model/age_bucketized/weights/part_0/AssignGlinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/age_bucketized/weights  "(2Elinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros:08
?
Llinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0:0Qlinear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/Assign`linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/Read/ReadVariableOp:0"Q
Clinear/linear_model/age_bucketized_X_education_X_occupation/weights  "(2^linear/linear_model/age_bucketized_X_education_X_occupation/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/education/weights/part_0:03linear/linear_model/education/weights/part_0/AssignBlinear/linear_model/education/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/education/weights  "(2@linear/linear_model/education/weights/part_0/Initializer/zeros:08
?
;linear/linear_model/education_X_occupation/weights/part_0:0@linear/linear_model/education_X_occupation/weights/part_0/AssignOlinear/linear_model/education_X_occupation/weights/part_0/Read/ReadVariableOp:0"@
2linear/linear_model/education_X_occupation/weights  "(2Mlinear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros:08
?
3linear/linear_model/marital_status/weights/part_0:08linear/linear_model/marital_status/weights/part_0/AssignGlinear/linear_model/marital_status/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/marital_status/weights  "(2Elinear/linear_model/marital_status/weights/part_0/Initializer/zeros:08
?
/linear/linear_model/occupation/weights/part_0:04linear/linear_model/occupation/weights/part_0/AssignClinear/linear_model/occupation/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/occupation/weights  "(2Alinear/linear_model/occupation/weights/part_0/Initializer/zeros:08
?
1linear/linear_model/relationship/weights/part_0:06linear/linear_model/relationship/weights/part_0/AssignElinear/linear_model/relationship/weights/part_0/Read/ReadVariableOp:0"6
(linear/linear_model/relationship/weights  "(2Clinear/linear_model/relationship/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/workclass/weights/part_0:03linear/linear_model/workclass/weights/part_0/AssignBlinear/linear_model/workclass/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/workclass/weights  "(2@linear/linear_model/workclass/weights/part_0/Initializer/zeros:08
?
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08*?
classification?
3
inputs)
input_example_tensor:0?????????-
classes"
head/Tile:0?????????A
scores7
 head/predictions/probabilities:0?????????tensorflow/serving/classify*?
predict?
5
examples)
input_example_tensor:0??????????
all_class_ids.
head/predictions/Tile:0??????????
all_classes0
head/predictions/Tile_1:0?????????A
	class_ids4
head/predictions/ExpandDims:0	?????????@
classes5
head/predictions/str_classes:0?????????>
logistic2
head/predictions/logistic:0?????????&
logits
add:0?????????H
probabilities7
 head/predictions/probabilities:0?????????tensorflow/serving/predict*?

regression?
3
inputs)
input_example_tensor:0?????????=
outputs2
head/predictions/logistic:0?????????tensorflow/serving/regress*?
serving_default?
3
inputs)
input_example_tensor:0?????????-
classes"
head/Tile:0?????????A
scores7
 head/predictions/probabilities:0?????????tensorflow/serving/classify