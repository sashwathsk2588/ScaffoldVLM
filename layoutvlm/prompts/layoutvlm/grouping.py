grouping_v0 = """
As a 3D layout designer, your task is to organize a list of assets for a 3D scene by grouping them based on their semantic or functional relationships.
These groups will guide the sequential placement of the assets within the scene. Ensure that each group consists of logically related assets, even if they are not physically adjacent, as this will help in generating a scene graph with constraint functions later on.

When sequencing the asset groups, prioritize the most significant or defining elements of the scene. For example, in a classroom, the whiteboard should be placed first to establish the front of the room, while in a bedroom, the bed might be the central piece to place initially. Generally, larger assets tend to be placed first.
Remember, you can include the same asset in multiple groups, but the position and rotation of an asset within a group should be uniquely defined by its relationship with other assets in that group.

It is essential to include all the assets and adhere to the room layout instructions provided when generating layout guidance for each semantic asset group.

Your objective is to deliver a clear and detailed organization of the 3D assets into logical groups, aligned with the given instructions on their placement. Make sure that the semantic asset groups are well-defined and conducive to sequential placement within the scene.

- Consider group the assets that are physically close to each other if they form a functional group or are semantically related.

- Try not to ouput a group with only one asset, unless there are very few objects in the room and they are not semantically related.

NOTE: it is very important to include all the assets!!!
Now, please proceed by organizing the following list of assets into logical groups for sequential placement. 

TASK: TASK_DESCRIPTION
LAYOUT INSTRUCTION: LAYOUT_CRITERIA

The 3D assets to be placed are described as following:
{asset_lists}

When you are generating layout instructions for each semantic asset group, please make sure to follow the room layout instruction given.
For example, given a classroom, you can define the first semantic asset group that consists of the whiteboard and and the desks in the scene.
The second semantic asset group could be all the desks and chairs, and laptops to be placed on top of the desks (note that the desks are also in the first semantic asset group but that is fine)
The third semantic asset group could be the lounge area towards the back of the classroom: vases, bookshelves, floor lamps next to lounge chairs.
The fourth asset group could be all the small objects you want to place on top of the students' desks or the books you want to place on bookshelves.
"""

grouping_v1_flat_text = """
You are an experienced 3D layout designer. You are teaching a junior designer the concept of semantic asset group. Understanding and recognizing semantic asset groups will help the designers to design a room layout more efficiently by decomposing the room layout into smaller, more manageable parts.

**Definition:**
A semantic asset group is a collection of assets that are logically related to each other. The grouping of the assets are based on functional, semantic, geometric, and functional relationships between assets.
Usually assets that are close to each other in space can be grouped together. For example, a bed, a nightstand, and a lamp on top of the nightstand can be grouped together.
However, it's also possible to group assets that are not physically close to each other but are semantically related. For example, a sofa, a tv console in front of the sofa, and a tv on top of the tv console can be grouped together even though the tv and the tv console is a few meters away from the sofa. They can be grouped together because they are semantically related -- the tv is in front of the sofa.

**Task:**
Now, given a 3D scene, you will use it as an example to teach the junior designer how to group assets into semantic asset groups.

**Step-by-Step Guide:**
1. You will first be provided a list of assets. Based on the assets, you should describe the general layout of the scene, the types of assets present, and any notable features.
2. You will identify the semantic relationships between the assets. You should consider the functional, semantic, and geometric relationships between the assets.
3. You will then describe how you would group the assets into semantic asset groups. You should explain the rationale behind each group and how the assets within each group are related to each other. 
4. You will then order the semantic asset groups based on the sequence in which they should be placed in the scene. You should consider the significance of each group and the logical flow of the scene layout. For example, larger or more prominent assets may be placed first to establish the scene's focal points.
5. Finally, you will format the grouping information into a clear and organized structure that can be easily understood by other designers or stakeholders.

**Example:**
Suppose you are examining a bedroom scene. In the bedroom, there are the following assets:
bed | ...
nightstand | ...
lamp | ...
bed_bench | ...
dresser-0 | ...
dresser-1 | ...
photo_frame-0 | ...
dressing_table-0 | ...
chair-0 | ...

1. After examining the scene, you will describe the scene a bedroom with a bed and a seating area for dressing.
2. You will list the assets and their relationships: 
- the bed is the central piece
- the nightstand is next to the bed for placing items. The bedside table should be close to the bed for easy access.
- the lamp is on the nightstand for lighting. The lamp should be close to the bed for reading.
- the end of bed bench is at the foot of the bed. The bench is at the end of the bed for seating or placing items.
- the dresser is on the other side of the room. The dresser is on the opposite side of the bed for storage.
- the photo frame is on the dresser. The photo is directly opposite the bed for viewing.
- the dressing table is in an open area of the room. 
- the chair is in front of the dressing table for seating.
3. You will group the assets into semantic asset groups:
- Group 1: Bed, Nightstand, Lamp. The rational is that the bed is the central piece, the nightstand is next to the bed, and the lamp is on the nightstand. They are related to each other because they are used for sleeping and reading.
- Group 2: End of bed bench. The bench is at the foot of the bed for seating or placing items.
- Group 3: Dresser, Photo frame. The dresser is on the opposite side of the bed for storage, and the photo frame is directly opposite the bed for viewing.
- Group 4: Dressing table, Chair. The dressing table is in an open area of the room, and the chair is in front of the dressing table for seating.
4. You will order the semantic asset groups based on the sequence in which they should be placed in the scene:
- Group 1: Bed, Nightstand, Lamp. They should be placed first to establish the sleeping area. They are the focal point of the room.
- Group 2: End of bed bench. It should be placed next to the bed to complement the sleeping area.
- Group 3: Dresser, Photo frame. They should be placed on the opposite side of the room to balance the layout.
- Group 4: Dressing table, Chair. They should be placed in an open area of the room to create a dressing area.
5. You will format the grouping information into a clear and organized structure:
```json
{
    "list": [
        {"id": 1, 
         "name": "sleeping area", 
         "assets": ["bed", "nightstand", "lamp"], 
         "rational": "they are used for sleeping and reading.",
         "key_relations_between_assets": ["the bed is the central piece", "the nightstand is next to the bed", "the lamp is on the nightstand"],
         "key_relations_with_other_groups": []
        },
        {"id": 2,
         "name": "seating area",
         "assets": ["bed_bench"],
         "rational": "this end of bed bench is at the foot of the bed for seating or placing items.",
         "key_relations_between_assets": [],
         "key_relations_with_other_groups": ["the bench complements the sleeping area."]
        },
        {"id": 3,
         "name": "storage area",
         "assets": ["dresser-0", "dresser-1", "photo_frame-0"],
         "rational": "the dresser is on the opposite side of the bed for storage, and the photo frame is directly opposite the bed for viewing.",
         "key_relations_between_assets": ["the photo frame is on top of the dresser"],
         "key_relations_with_other_groups": ["the dresser is for storage, and the photo frame is for viewing. To make the photo frame visible from the bed, the dresser should be placed on the opposite side of the bed."]
        },
        {"id": 4,
         "name": "dressing area",
         "assets": ["dressing_table-0", "chair-0"],
         "rational": "the dressing table is in an open area of the room, and the chair is in front of the dressing table for seating.",
         "key_relations_between_assets": ["the chair is in front of the dressing table"],
         "key_relations_with_other_groups": ["the dressing area complements the sleeping area and the storage area by providing another function in the room."]
        }
    ]
}


```

Now, please proceed by grouping and organizing the following list of assets according to the layout instruction:
NOTE: it is very important to include all the assets!!! And please do not change the name of the assets.

Task: TASK_DESCRIPTION
Instruction: LAYOUT_CRITERIA
In the room, there are the following assets:
ASSET_LISTS
"""

grouping_v1_flat = """
You are an experienced 3D layout designer. You are teaching a junior designer the concept of semantic asset group. Understanding and recognizing semantic asset groups will help the designers to design a room layout more efficiently by decomposing the room layout into smaller, more manageable parts.

**Definition:**
A semantic asset group is a collection of assets that are logically related to each other. The grouping of the assets are based on functional, semantic, geometric, and functional relationships between assets.
Usually assets that are close to each other in space can be grouped together. For example, a bed, a nightstand, and a lamp on top of the nightstand can be grouped together.
However, it's also possible to group assets that are not physically close to each other but are semantically related. For example, a sofa, a tv console in front of the sofa, and a tv on top of the tv console can be grouped together even though the tv and the tv console is a few meters away from the sofa. They can be grouped together because they are semantically related -- the tv is in front of the sofa.

**Task:**
Now, given a 3D scene, you will use it as an example to teach the junior designer how to group assets into semantic asset groups.

**Step-by-Step Guide:**
1. You will first examine one or multiple images of the 3D scene from different angles. You should describe the general layout of the scene, the types of assets present, and any notable features.
2. You will identify the assets in the scene and the relationships between them. Each asset should be assigned a unique identifier based on the annotated image (e.g., Side Table-0). You should consider the functional, semantic, and geometric relationships between the assets.
3. You will then describe how you would group the assets into semantic asset groups. You should explain the rationale behind each group and how the assets within each group are related to each other. 
4. You will then order the semantic asset groups based on the sequence in which they should be placed in the scene. You should consider the significance of each group and the logical flow of the scene layout. For example, larger or more prominent assets may be placed first to establish the scene's focal points.
5. Finally, you will format the grouping information into a clear and organized structure that can be easily understood by other designers or stakeholders.

**Example:**
Suppose you are examining a bedroom scene. In the bedroom, there are the following assets:
bed | ...
nightstand | ...
lamp | ...
bed_bench | ...
dresser-0 | ...
dresser-1 | ...
photo_frame-0 | ...
dressing_table-0 | ...
chair-0 | ...

1. After examining the scene, you will describe the scene a bedroom with a bed and a seating area for dressing.
2. You will list the assets and their relationships: 
- the bed is the central piece
- the nightstand is next to the bed for placing items. The bedside table should be close to the bed for easy access.
- the lamp is on the nightstand for lighting. The lamp should be close to the bed for reading.
- the end of bed bench is at the foot of the bed. The bench is at the end of the bed for seating or placing items.
- the dresser is on the other side of the room. The dresser is on the opposite side of the bed for storage.
- the photo frame is on the dresser. The photo is directly opposite the bed for viewing.
- the dressing table is in an open area of the room. 
- the chair is in front of the dressing table for seating.
3. You will group the assets into semantic asset groups:
- Group 1: Bed, Nightstand, Lamp. The rational is that the bed is the central piece, the nightstand is next to the bed, and the lamp is on the nightstand. They are related to each other because they are used for sleeping and reading.
- Group 2: End of bed bench. The bench is at the foot of the bed for seating or placing items.
- Group 3: Dresser, Photo frame. The dresser is on the opposite side of the bed for storage, and the photo frame is directly opposite the bed for viewing.
- Group 4: Dressing table, Chair. The dressing table is in an open area of the room, and the chair is in front of the dressing table for seating.
4. You will order the semantic asset groups based on the sequence in which they should be placed in the scene:
- Group 1: Bed, Nightstand, Lamp. They should be placed first to establish the sleeping area. They are the focal point of the room.
- Group 2: End of bed bench. It should be placed next to the bed to complement the sleeping area.
- Group 3: Dresser, Photo frame. They should be placed on the opposite side of the room to balance the layout.
- Group 4: Dressing table, Chair. They should be placed in an open area of the room to create a dressing area.
5. You will format the grouping information into a clear and organized structure:
```json
{
    "list": [
        {"id": 1, 
         "name": "sleeping area", 
         "assets": ["bed", "nightstand", "lamp"], 
         "rational": "they are used for sleeping and reading.",
         "key_relations_between_assets": ["the bed is the central piece", "the nightstand is next to the bed", "the lamp is on the nightstand"],
         "key_relations_with_other_groups": []
        },
        {"id": 2,
         "name": "seating area",
         "assets": ["bed_bench"],
         "rational": "this end of bed bench is at the foot of the bed for seating or placing items.",
         "key_relations_between_assets": [],
         "key_relations_with_other_groups": ["the bench complements the sleeping area."]
        },
        {"id": 3,
         "name": "storage area",
         "assets": ["dresser-0", "dresser-1", "photo_frame-0"],
         "rational": "the dresser is on the opposite side of the bed for storage, and the photo frame is directly opposite the bed for viewing.",
         "key_relations_between_assets": ["the photo frame is on top of the dresser"],
         "key_relations_with_other_groups": ["the dresser is for storage, and the photo frame is for viewing. To make the photo frame visible from the bed, the dresser should be placed on the opposite side of the bed."]
        },
        {"id": 4,
         "name": "dressing area",
         "assets": ["dressing_table-0", "chair-0"],
         "rational": "the dressing table is in an open area of the room, and the chair is in front of the dressing table for seating.",
         "key_relations_between_assets": ["the chair is in front of the dressing table"],
         "key_relations_with_other_groups": ["the dressing area complements the sleeping area and the storage area by providing another function in the room."]
        }
    ]
}
```

Now, please proceed by grouping and organizing the following list of assets according to the layout instruction:
NOTE: it is very important to include all the assets!!! And please do not change the name of the assets.

In the room, there are the following assets:
ASSET_LISTS
"""