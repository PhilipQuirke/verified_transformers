import unittest

from QuantaTools.model_token_to_char import token_to_char, tokens_to_string

from QuantaTools.useful_node import UsefulNode, UsefulNodeList

from QuantaTools.quanta_constants import QType
from QuantaTools.quanta_map_impact import sort_unique_digits


class TestUseful(unittest.TestCase):


    def test_useful_node(self):
        
        useful_node = UsefulNode(1, 2, True, 3, [])

        # Add 6 distinct tags
        useful_node.add_tag( "Major0", "Minor1" )
        useful_node.add_tag( "Major0", "Minor2" )
        useful_node.add_tag( "Major1", "Minor1" )
        useful_node.add_tag( "Major1", "Minor2" )
        useful_node.add_tag( "Major2", "Minor1" )
        useful_node.add_tag( "Major2", "Minor2" )
        self.assertEqual( len(useful_node.tags), 6)
        
        # Add 2 duplicate tags
        useful_node.add_tag( "Major1", "Minor1" )
        useful_node.add_tag( "Major2", "Minor2" )
        self.assertEqual( len(useful_node.tags), 6)

        # Remove 2 tags
        useful_node.reset_tags( "Major2" )
        self.assertEqual( len(useful_node.tags), 4)

        # Remove 1 tag
        useful_node.reset_tags( "Major0", "Minor2" )
        self.assertEqual( len(useful_node.tags), 3)

        # Remove all tags
        useful_node.reset_tags( "" )
        self.assertEqual( len(useful_node.tags), 0)

