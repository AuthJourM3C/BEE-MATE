a = {'dog': 0.45, 'cow': 0.1, 'speech': 0.35, 'sea waves': 0.05, 'engine': 0.05}
print (list(a.keys()))

ESC_50_pollution_classes = [ 'washing machine', 'vacuum cleaner', 'hellicopter', 'chainsaw', 'siren', 'car horn', 'engine', 'train','airplane']

for key in list(a.keys()):
    if key in ESC_50_pollution_classes:
        print ("it's polluting")

