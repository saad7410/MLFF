from netCDF4 import Dataset
ds = Dataset("../data/A03_53310/A03_butene_0p50fs_dynamic.nc")
print("VARS:", list(ds.variables.keys()))
print("GLOBAL ATTRS:", ds.ncattrs())
for k,v in ds.variables.items():
    print(k, v.shape, v.dtype, v.ncattrs())
