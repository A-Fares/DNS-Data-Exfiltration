import numpy as np
import collections
import tldextract
import pandas as pd


def get_FQDN_count(domainName):
    """Returns the length of domain Name
       Args:
         domainName: a domain name string
       """
    word_char = [(ch) for ch in domainName]
    return len(word_char)


def get_subdomain_length(domainName):
    """Returns the length of the subdomain
       Args:
         domainName: a domain name string
       """
    le = []
    for i in domainName:
        sub_domain, domain, suffix = tldextract.extract(i)
        length = len(sub_domain)
        le.append(length)
    return le


def get_upper_count(domainName):
    """Returns the count of upper characters in the domain
       Args:
         domainName: a domain name string
       """
    count = 0
    for char in domainName:
        if char.isupper():
            count += 1
    return count


def get_lower_count(domainName):
    """Returns the count of lower characters in the domain
       Args:
         domainName: a domain name string
       """
    count = 0
    for char in domainName:
        if char.islower():
            count += 1
    return count


def get_numeric_count(domainName):
    """Returns the count of numeric values in the domain
       Args:
         domainName: a domain name string
       """
    count = 0
    for char in domainName:
        if char.isnumeric():
            count += 1
    return count


def get_entropy(domain):
    """Returns the entropy of the domain
         Args:
           domainName: a domain name string
         """
    p, lengths = collections.Counter(domain), np.float(len(domain))
    return -np.sum(count / lengths * np.log2(count / lengths) for count in p.values())


def get_special_count(domain_ch):
    """Returns the special characters count
         Args:
           domainName: a domain name string
         """
    count = 0
    for char in domain_ch:
        char_int = ord(char)
        if ((32 <= char_int <= 47) or (58 <= char_int <= 64) or
                (91 <= char_int <= 96) or (123 <= char_int <= 126)):
            count += 1
    return count


def get_labels(domainName):
    """Returns the labels chunks
         Args:
           domainName: a domain name string
         """
    chunks = domainName.split(".")
    return len(chunks)


def get_labels_max(domainName):
    """Returns the max label length
         Args:
           domainName: a domain name string
         """
    chunks = domainName.split(".")
    max_length = max([(len(x)) for x in chunks])
    return max_length


def get_labels_average(domainName):
    """Returns the average of labels count
         Args:
           domainName: a domain name string
         """
    chunks = domainName.split(".")
    total_length = sum([(len(x)) for x in chunks])
    average_word = total_length / len(chunks)
    return average_word


def get_longest_word(domainName):
    """Returns the longest word in the domain name
         Args:
           domainName: a domain name string
         """
    chunks = domainName.split(".")
    longest_elements = [(len(x)) for x in chunks]
    longest_element = longest_elements.index(max(longest_elements))
    return chunks[longest_element]


def get_sld(domainName):
    """Returns the second level domain
         Args:
           domainName: a domain name string
         """
    sld = [tldextract.extract(i).domain for i in domainName]
    return sld


def get_len(domainName):
    """Returns the length of subdomain and domain
         Args:
           domainName: a domain name string
         """
    lengths = []
    for i in domainName:
        sub_domain, domain, suffix = tldextract.extract(i)
        le = len(sub_domain) + len(domain)
        lengths.append(le)
    return lengths


def is_subdomain(domainName):
    """Returns the flag that check if there is a subdomain or not
         Args:
           domainName: a domain name string
         """
    subdomain = tldextract.extract(domainName).subdomain
    if subdomain:
        return 1
    else:
        return 0


def build_features(domain_df):
    """
      returns a dataframe with 14 features

      Args:
          df: A dataframe with the domain column

      Returns:
          A dataframe with 14 columns features
      """
    domainName = domain_df['domain']
    data = {"FQDN_count": domainName.apply(get_FQDN_count),
            "subdomain_length": get_subdomain_length(domainName=domainName),
            "upper_count": domainName.apply(get_upper_count),
            "lower_count": domainName.apply(get_lower_count),
            "numeric_count": domainName.apply(get_numeric_count),
            "entropy": domainName.apply(get_entropy),
            "special_count": domainName.apply(get_special_count),
            "labels": domainName.apply(get_labels),
            "get_labels_max": domainName.apply(get_labels_max),
            "labels_average": domainName.apply(get_labels_average),
            "longest_word": domainName.apply(get_longest_word),
            "sld": get_sld(domainName=domainName),
            "len": get_len(domainName=domainName),
            "subdomain": domainName.apply(is_subdomain)
            }

    df = pd.DataFrame(data)
    return df


def preprocessing_data(dataframe):
    """
          Args:
              df: A dataframe

          Returns:
              A preprocessing dataframe
          """
    categorical_column = [i for i in dataframe.columns if dataframe[i].dtype == "object"]
    for cat in categorical_column:
        dataframe = dataframe.drop(cat, axis=1)
    return dataframe
