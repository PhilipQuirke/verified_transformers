def row_location_name(layer, is_head, num):
  return "L" + str(layer) + ("H" + str(num) if is_head else "MLP")


def location_name(position, layer, is_head, num):
  return "P" + str(position) + row_location_name(layer, is_head, num)


# The unique location of a node (attention head or MLP neuron) in a model 
class NodeLocation():
  position : int  # Token position 
  layer : int
  is_head : bool # Else is MLP neuron
  num: int # Either attention head or MLP neuron number

  def __init__(self, position, layer, is_head, num):
    self.position = position
    self.layer = layer
    self.is_head = is_head
    self.num = num


  # Node name e.g. P14L2H3
  def name(self):
    return location_name(self.position,self.layer,self.is_head,self.num)


  def row_name(self):
    return row_location_name(self.layer,self.is_head,self.num)
    

class UsefulNode(NodeLocation):

  # Tags related to the node of form "MajorVersion:MinorVersion"
  tags : list


  def __init__(self, position, layer, is_head, num):
    super().__init__(position, layer, is_head, num)  # Call the parent class's constructor    
    self.tags = []

  
  # Remove some/all tags from this 
  def reset_tags(self, major_tag):
    if major_tag == "":
      self.tags = []
    else:
      self.tags = [s for s in self.tags if not s.startswith(major_tag)]


  # Add a tag to this  (if not already present)
  def add_tag(self, major_tag, minor_tag):
    tag = major_tag + ":" + minor_tag
    if tag != "" and (not (tag in self.tags)):
      self.tags += [tag]


  # Return tags with the matching major and minor versions
  def filter_tags(self, major_tag, minor_tag = ""):
    assert major_tag != ""

    filtered_strings = [s for s in self.tags if s.startswith(major_tag)]

    minor_tags = [s.split(":")[1] for s in filtered_strings]

    if minor_tag != "":
      minor_tags = [s for s in minor_tags if s.startswith(minor_tag)]

    return minor_tags


  # Return minimum tag with the matching major and minor versions
  def min_tag_suffix(self, major_tag, minor_tag = ""):
    assert major_tag != ""

    minor_tags = self.filter_tags(major_tag)

    if minor_tag != "":
      minor_tags = [s for s in minor_tags if s.startswith(minor_tag)]

    return min(minor_tags) if minor_tags else ""


  # Return the only tag with the matching major_tag
  def only_tag(self, major_tag):
    assert major_tag != ""

    filtered_strings = [s for s in self.tags if s.startswith(major_tag)]

    num_strings = len(filtered_strings)
    if num_strings > 1:
      print("only_tag logic failure", major_tag, num_strings, filtered_strings)
      assert False

    return filtered_strings[0].split(":")[1] if num_strings == 1 else ""


  # Return whether this  contains a tag with the matching major_tag or major+minor_tag
  def contains_tag(self, major_tag, minor_tag):
    assert major_tag != ""

    for tag in self.tags:
      # We use contains(minor) as the POSITION_MAJOR_TAG minor tag is "P14=25%"
      if tag.startswith(major_tag) and minor_tag in tag:
        return True

    return False


  def to_dict(self):
    return {
      "position": self.position,
      "layer": self.layer,
      "is_head": self.is_head,
      "num": self.num,
      "tags": self.tags
    }
