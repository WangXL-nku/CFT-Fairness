import os


# 设置文件路径
base_save_path = '/media/data0/wzy/'
attr_file_path = os.path.join(base_save_path, 'dataset/CelebA/Anno/list_attr_celeba.txt')

# 读取属性文件
def read_attributes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        attributes = [line.strip().split() for line in lines[2:]]  # 跳过前两行
        attr_name = lines[1].strip().split()
    return attributes, attr_name

# 计算特定属性在男性和女性中的比例
def calculate_attribute_proportions(attributes, check_tid, tid):
    male_count = 0
    female_count = 0
    male_attribute_count = 0
    female_attribute_count = 0
    total_count = 0


    for attr in attributes:
        total_count += 1
        gender = attr[check_tid]
        if gender == '1':
            male_count += 1
            if attr[tid] == '1':
                male_attribute_count += 1
        elif gender == '-1':  # 假设'0'代表女性
            female_count += 1
            if attr[tid] == '1':
                female_attribute_count += 1

    # 计算比例
    male_proportion = male_attribute_count / male_count if male_count > 0 else 0
    female_proportion = female_attribute_count / female_count if female_count > 0 else 0

    return male_proportion, female_proportion

# 主函数
def main():
    check_tid = 10
    attributes, attr_name = read_attributes(attr_file_path)
    for tid in range(len(attr_name)):
        attribute_name = attr_name[tid-1] # 可以替换为其他属性名称
        male_proportion, female_proportion = calculate_attribute_proportions(attributes, check_tid, tid=tid)
        print("__________________________________")
        print(f"{attribute_name}在{attr_name[check_tid-1]}中的比例: {male_proportion:.2f}")
        print(f"NO-{attribute_name}在NO-{attr_name[check_tid-1]}中的比例: {female_proportion:.2f}")

if __name__ == "__main__":
    main()