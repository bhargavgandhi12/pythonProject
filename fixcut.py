#Extract list of FIX tags provided by the user from the Fix message.
#Files takes arguments as list of tags from the user
#Sample File : Read from FixMessage.txt
#Sending Fix Message|8=FIX4.29=5634=143=N35=D49=SENDER56=TARGET55=IBM11=ClientOrderId138=10054=140=244=12359=0115=CLIENT116=TRADERID100=XNDQ108=30123=N52=20231212-11:23:33.56710=554
#Received Fix Message|8=FIX4.29=4434=243=N35=849=TARGET56=SENDER55=IBM11=ClietOrderId137=BrokerOrderId138=10054=140=244=12359=0128=CLIENT129=TRADERID76=XNDQ39=0150=0108=30453=1448=BRK2447=D452=152=20231212-11:23:33.56710=33

from pprint import pprint


def process_fix_file(file: str, tags: list):
    """
    Define function that opens the file , Read lines from file and takes tags as arguments to extract as list
    """
    with open(file, 'r') as file:
        if not file:
            return "File Not Found!"
        else:
            for line in file:
                line = line.strip()
                fix_tags(line, tags)


def fix_tags(fixmsg: str, tags: list = None):
    """
    Read each line and Extract tags. tag11 and 37 is required in the message to create unique keys for dict of Incoming and Outgoing messages
    """
    extracted_fixmsg = {}
    tag_value_dict = {}
    if fixmsg:
        direction_fixmsg = fixmsg.split('|')[0]
        msg_with_tags = fixmsg.split('|')[1]
        tag_value_list = msg_with_tags.split(sep='\x01')
        for tag_pair in tag_value_list:
            tag = tag_pair.split(sep='=')[0]
            value = tag_pair.split(sep='=')[1]
            tag_value_dict[tag] = value
            if tag == '11':
                clientorder_id = tag_value_dict['11']
            if tag == '37':
                broker_id = tag_value_dict['37']
        if 'Sending Fix Message' in direction_fixmsg and tag_value_dict['11']:
            extracted_fixmsg[direction_fixmsg + '_' + clientorder_id] = tag_value_dict
        elif 'Received Fix Message' in direction_fixmsg and tag_value_dict['37']:
            extracted_fixmsg[direction_fixmsg + '_' + broker_id] = tag_value_dict
    if tags is not None:
        print_selective_tags(extracted_fixmsg, tags)
    print(extracted_fixmsg)


def print_selective_tags(extracted_fixmsg: dict, ip_tags: list):
    """
    Based on tags provided by user create new dict for display from original extracted_fixmsg
    :param extracted_fixmsg:
    :param ip_tags:
    :return:
    """
    selective_tags_dict = {}
    selective_tag_val_pair = {}
    for k in extracted_fixmsg.keys():
        selective_tags_dict[k] = None
        for key, value in extracted_fixmsg.items():
            if isinstance(value, dict):
                for orig_tag, orig_val in value.items():
                    for ip_tag in ip_tags:
                        if orig_tag == str(ip_tag):
                            selective_tag_val_pair.update({orig_tag: orig_val})
            selective_tags_dict[key] = selective_tag_val_pair
            selective_tag_val_pair = {}

    pprint(selective_tags_dict)

    # print(extracted_fixmsg)


process_fix_file('FixMessage.txt', [35, 49, 56, 55, 57, 50, 54, 40, 52, 60])

### Output dict : { 'Sending Fix Message ClientOrderId' :
#                       { '35': 'D'
#                         '49': 'SENDER'
#                         '56': 'TARGET'
#                       }
#                  }
#                 { 'Received Fix Message ClientOrderId_BrokerId' :
#                       { '35': '8'
#                         '56': 'SENDER'
#                         '49': 'TARGET'
#                       }
#                  }
