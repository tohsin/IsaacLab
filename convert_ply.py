import argparse
import asyncio
from isaacsim import SimulationApp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PLY to USD using Omniverse Asset Converter")
    parser.add_argument("input_ply", type=str, help="Path to the input PLY file")
    parser.add_argument("output_usd", type=str, help="Path to the output USD file")
    args = parser.parse_args()

    # Start the Omniverse standalone application
    simulation_app = SimulationApp({"headless": True})

    from omni.isaac.core.utils.extensions import enable_extension
    
    # Enable the asset converter extension
    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter

    async def convert_ply_to_usd(input_ply_path: str, output_usd_path: str):
        converter_context = omni.kit.asset_converter.AssetConverterContext()
        # You can configure settings like up-axis, units, etc., here
        # converter_context.setup_default(bExportHidden=False)
        
        instance = omni.kit.asset_converter.get_instance()
        print(f"Converting {input_ply_path} to {output_usd_path}...")
        
        task = instance.create_converter_task(input_ply_path, output_usd_path, None, converter_context)
        success = await task.wait_until_finished()
        
        if success:
            print(f"Conversion successful! Saved to {output_usd_path}")
        else:
            print("Conversion failed.")

    # Run the async loop
    asyncio.get_event_loop().run_until_complete(
        convert_ply_to_usd(args.input_ply, args.output_usd)  
    )

    simulation_app.close()
