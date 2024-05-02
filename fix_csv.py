import csv

def fix_timesteps(input_file, output_file):
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        save_timestep = -1
        last_timestep = -1
        for row in rows:
            timestep = int(row[0])
            if timestep < last_timestep:
                if timestep == 2560:
                    save_timestep = last_timestep
                timestep += save_timestep
                row[0] = str(timestep)
            else:
                save_timestep = timestep
            last_timestep = int(row[0])
            writer.writerow(row)

# 使用例
fix_timesteps('log/24-04-17 18-45-23/train/reward.csv', 'log/24-04-17 18-45-23/train/reward_.csv')
print("finish")