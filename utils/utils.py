import yaml
from easydict import EasyDict
import os

def merge_dicts(base, override):
    """
    override 覆盖 base（递归）
    """
    for k, v in override.items():
        if isinstance(v, dict) and k in base:
            base[k] = merge_dicts(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path, merge_data_config=True, strict_check=True):
    """
    支持 data_config 继承，并检查 data 和 model 的类别数一致性

    Args:
        config_path (str): 配置文件路径
        merge_data_config (bool): 是否合并 data_config，默认为 True
        strict_check (bool): 是否进行严格检查。如果为 True：
                             - 两者都存在但不一致 -> 抛出错误
                             - 只有一个存在 -> 抛出错误
                             如果为 False：
                             - 两者都存在但不一致 -> 抛出错误
                             - 只有一个存在 -> 仅警告

    Returns:
        EasyDict: 配置字典
    """
    # 确保 config_path 是绝对路径
    if not os.path.isabs(config_path):
        # 假设 config_path 相对于当前工作目录
        config_path = os.path.abspath(config_path)

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 如果有 data_config
    if merge_data_config and 'data_config' in config:
        data_config_path = config['data_config']

        # 清理路径字符串（去除可能的引号和空格）
        data_config_path = data_config_path.strip().strip('\'"')

        # 处理相对路径
        if not os.path.isabs(data_config_path):
            # 相对于主配置文件所在目录
            config_dir = os.path.dirname(config_path)
            data_config_path = os.path.join(config_dir, data_config_path)
            # 规范化路径（处理 .. 和 .）
            data_config_path = os.path.normpath(data_config_path)

        # 检查文件是否存在
        if not os.path.exists(data_config_path):
            raise FileNotFoundError(
                f"Data config file not found: {data_config_path}\n"
                f"Resolved from: config['data_config'] = {config['data_config']}\n"
                f"Main config path: {config_path}"
            )

        print(f"Loading data config from: {data_config_path}")

        with open(data_config_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        config = merge_dicts(data_config, config)

    # 转换为 EasyDict 以便于访问
    config = EasyDict(config)

    # 检查 data 和 model 的类别数一致性
    _check_num_classes_consistency(config, strict_check)

    return config

def _check_num_classes_consistency(config, strict_check=True):
    """
    检查 data.num_classes 和 model.params.num_classes 是否一致
    """
    # 获取 data 中的类别数
    data_num_classes = None
    if hasattr(config, 'data') and hasattr(config.data, 'num_classes'):
        data_num_classes = config.data.num_classes

    # 获取 model 中的类别数
    model_num_classes = None
    if (hasattr(config, 'model') and
            hasattr(config.model, 'params') and
            hasattr(config.model.params, 'num_classes')):
        model_num_classes = config.model.params.num_classes

    # 如果两者都存在，检查一致性
    if data_num_classes is not None and model_num_classes is not None:
        if data_num_classes != model_num_classes:
            error_msg = (
                f"\n{'=' * 60}\n"
                f"Category count inconsistency error:\n"
                f"  data.num_classes = {data_num_classes}\n"
                f"  model.params.num_classes = {model_num_classes}\n"
                f"\nPlease ensure that the number of categories in the data configuration matches the number of categories in the model configuration!\n"
                f"{'=' * 60}"
            )
            raise ValueError(error_msg)
        else:
            print(f"✓ Category count check passed: data.num_classes = model.params.num_classes = {data_num_classes}")

    # 如果只有一个存在
    elif data_num_classes is None and model_num_classes is not None:
        msg = f"model.params.num_classes = {model_num_classes}，但 data 配置中没有指定 num_classes"
        if strict_check:
            raise ValueError(f"严格检查模式下，{msg}")
        else:
            print(f"⚠️ 警告: {msg}")

    elif data_num_classes is not None and model_num_classes is None:
        msg = f"data.num_classes = {data_num_classes}，但 model 配置中没有指定 num_classes"
        if strict_check:
            raise ValueError(f"严格检查模式下，{msg}")
        else:
            print(f"⚠️ 警告: {msg}")

def save_merged_config(config, save_dir):
    save_path = os.path.join(save_dir, f'{config.model.name}.yaml')

    # EasyDict → dict
    def to_dict(obj):
        if isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            to_dict(config),
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False  # 控制是否用 {a:1}
        )

    print(f"The configuration has been saved to: {save_path}")

if __name__=="__main__":
    import pyzjr
    pyzjr.show_config(yaml_path=r'E:\PythonProject\CDLab\config\MSCANet.yaml')
    conf = load_config(r'E:\PythonProject\CDLab\config\MSCANet.yaml')
    print(conf.data.color_map)
    save_merged_config(conf, './')