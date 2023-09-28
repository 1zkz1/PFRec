# read config
import yaml

def get_config_data(yaml_file):
    # open yaml file
    print('######## get config info from ', yaml_file, '########')

    file = open(yaml_file, 'r', encoding='utf-8')
    file_data = file.read()
    file.close()

    # print(file_data)
    # convert to dict
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    # print(data)
    return data

if __name__ == "__main__":
    yaml_file = 'config.yaml'
    get_config_data(yaml_file)
    print()

