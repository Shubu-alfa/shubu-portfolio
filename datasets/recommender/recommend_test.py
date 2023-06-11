from Product_recommmendation import get_recommend

input_item = input("Enter the item already bought:")

recommended_item = get_recommend(input_item)

print("Recommended_item:", recommended_item)
