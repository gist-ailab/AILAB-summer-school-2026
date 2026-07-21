import h5py

src_file = "datasets/tbar_pickpalce_teleop_0719_final.hdf5"
dst_file = "datasets/tbar_pickpalce_teleop_0719_final_0_9.hdf5"

with h5py.File(src_file, "r") as src, h5py.File(dst_file, "w") as dst:
    dst_data = dst.create_group("data")
    
    for attr_name, attr_val in src["data"].attrs.items():
        if attr_name == "total":
            dst_data.attrs["total"] = 10
        else:
            dst_data.attrs[attr_name] = attr_val
            
    for i in range(10):
        demo_name = f"demo_{i}"
        if demo_name in src["data"]:
            src.copy(src["data"][demo_name], dst_data, name=demo_name)
        else:
            print(f"Warning: {demo_name} not found in source file.")
            
    # Check if env_args exists directly under root or data
    if "env_args" in src["data"]:
        src.copy(src["data"]["env_args"], dst_data, name="env_args")
    elif "env_args" in src:
        src.copy(src["env_args"], dst, name="env_args")

print(f"Successfully extracted demo 0~9 to {dst_file}")
