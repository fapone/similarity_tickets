import datetime
import ftfy
import re
import six
from unidecode import unidecode
import typing_utils

def _deserialize(data, klass):
    """Deserializes dict, list, str into an object.

    :param data: dict, list or str.
    :param klass: class literal, or string of class name.

    :return: object.
    """
    if data is None:
        return None

    if klass in six.integer_types or klass in (float, str, bool, bytearray):
        return _deserialize_primitive(data, klass)
    elif klass == object:
        return _deserialize_object(data)
    elif klass == datetime.date:
        return deserialize_date(data)
    elif klass == datetime.datetime:
        return deserialize_datetime(data)
    elif typing_utils.is_generic(klass):
        if typing_utils.is_list(klass):
            return _deserialize_list(data, klass.__args__[0])
        if typing_utils.is_dict(klass):
            return _deserialize_dict(data, klass.__args__[1])
    else:
        return deserialize_model(data, klass)


def _deserialize_primitive(data, klass):
    """Deserializes to primitive type.

    :param data: data to deserialize.
    :param klass: class literal.

    :return: int, long, float, str, bool.
    :rtype: int | long | float | str | bool
    """
    try:
        value = klass(data)
    except UnicodeEncodeError:
        value = six.u(data)
    except TypeError:
        value = data
    return value


def _deserialize_object(value):
    """Return an original value.

    :return: object.
    """
    return value


def deserialize_date(string):
    """Deserializes string to date.

    :param string: str.
    :type string: str
    :return: date.
    :rtype: date
    """
    if string is None:
      return None
    
    try:
        from dateutil.parser import parse
        return parse(string).date()
    except ImportError:
        return string


def deserialize_datetime(string):
    """Deserializes string to datetime.

    The string should be in iso8601 datetime format.

    :param string: str.
    :type string: str
    :return: datetime.
    :rtype: datetime
    """
    if string is None:
      return None
    
    try:
        from dateutil.parser import parse
        return parse(string)
    except ImportError:
        return string


def deserialize_model(data, klass):
    """Deserializes list or dict to model.

    :param data: dict, list.
    :type data: dict | list
    :param klass: class literal.
    :return: model object.
    """
    instance = klass()

    if not instance.openapi_types:
        return data

    for attr, attr_type in six.iteritems(instance.openapi_types):
        if data is not None \
                and instance.attribute_map[attr] in data \
                and isinstance(data, (list, dict)):
            value = data[instance.attribute_map[attr]]
            setattr(instance, attr, _deserialize(value, attr_type))

    return instance


def _deserialize_list(data, boxed_type):
    """Deserializes a list and its elements.

    :param data: list to deserialize.
    :type data: list
    :param boxed_type: class literal.

    :return: deserialized list.
    :rtype: list
    """
    return [_deserialize(sub_data, boxed_type)
            for sub_data in data]


def _deserialize_dict(data, boxed_type):
    """Deserializes a dict and its elements.

    :param data: dict to deserialize.
    :type data: dict
    :param boxed_type: class literal.

    :return: deserialized dict.
    :rtype: dict
    """
    return {k: _deserialize(v, boxed_type)
            for k, v in six.iteritems(data)}


def transform_sentence(sentence):
    # Ensure the parameter type as string
    mproc0 = str(sentence)

    # Set all messages to a standard encoding
    mproc1 = ftfy.fix_encoding(mproc0)

    # Replaces accentuation from chars. Ex.: "férias" becomes "ferias"
    mproc2 = unidecode(mproc1)

    # Removes special chars from the sentence. Ex.:
    #  - before: "MP - SIGAEST - MATA330/MATA331 - HELP CTGNOCAD"
    #  - after:  "MP   SIGAEST   MATA330 MATA331   HELP CTGNOCAD"
    mproc3 = re.sub('[^0-9a-zA-Z]', " ", mproc2)

    with open('custom_stopwords.txt', 'r') as file:
        custom_stopwords = file.read().splitlines()

    # Sets capital to lower case maintaining full upper case tokens and remove portuguese stop words.
    #  - before: "MP   MEU RH   Horario ou Data registrado errado em solicitacoes do MEU RH"
    #  - after:  "MP MEU RH horario data registrado errado solicitacoes MEU RH"
    mproc4 = " ".join([t.lower() for t in mproc3.split() if t not in custom_stopwords])

    return mproc4


def get_language_by_locale_code(locale_code: str) -> tuple[str, str]:
    """Returns the language name from the locale code.

    :param str: The locale code.
    :type str: str
    :return: The language name.
    :rtype: str
    """
    locale_code = locale_code.lower() if locale_code else ''
    if 'en' in locale_code:
        return 'English', 'en'
    elif 'es' in locale_code:
        return 'Spanish', 'es'
    return 'Brazilian Portuguese', 'pt'

