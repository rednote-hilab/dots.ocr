import asyncio
import os
import json
import re
from PIL import Image
from dots_ocr.utils.page_parser import PageParser
from dots_ocr.utils.consts import MAX_LENGTH, MAX_PIXELS

class CropImage:
    def __init__(self, image: Image, x_offset=0):
        self.image = image
        self.x_offset = x_offset

MAX_LEVEL=100
class SectionHeader:
    def __init__(self, text, category:str, bbox, level=None, source_block=None):
        self.text = text
        self.category = category
        self.bbox = bbox
        self.source_block = source_block

        self.level = level if level is not None else self._extract_level_from_text()
        self.new_level = None
        self.clean_text = self._clean_text()
        self.crop_img: CropImage = None
        
    @classmethod
    def from_info_block(cls, info_block, level=None):

        text = info_block.get('text', "")
        category = info_block['category']
        bbox = info_block['bbox']
        
        return cls(text, category, bbox, level=level, source_block=info_block)
    
    def _extract_level_from_text(self):
        """Extract header level from markdown-style text (# ## ### etc.)"""
        if self.category == 'Title':
            return 0

        hash_match = re.match(r'^(#{1,6})\s+', self.text)
        bold_match = re.search(r'\*\*(.*?)\*\*', self.text)
        tt = 8
        if hash_match:
            tt = len(hash_match.group(1))
        elif bold_match:
            tt = 7
            
        if self.category == 'Section-header':
            return tt
        elif self.category == 'List-item':
            return 10 + tt
        else:
            return 20 + tt
    
    def _clean_text(self):
        """Remove markdown symbols from text"""
        self.text = re.sub(r'^#{1,6}\s+', '', self.text)
        self.text = re.sub(r'^\*\*(.*?)\*\*$', r'\1', self.text.strip())
        return self.text
    
    def _reset_text_and_update(self):
        if self.new_level is None:
            return
        
        lines = self.clean_text.split('\n')
        formatted_lines = []

        for line in lines:
            if not line:
                continue
                
            if self.new_level == 1:
                self.category = "title"
                formatted_lines.append('# ' + line)
            elif self.new_level == 7:
                self.category = "Section-header"
                formatted_lines.append('**' + line + '**')
            elif self.new_level == 8:
                self.category = "list-item"
                formatted_lines.append(line)
            else:
                self.category = "Section-header"
                formatted_lines.append('#' * self.new_level + ' ' + line)
        
        self.text = '\n'.join(formatted_lines)

        if self.source_block:
            self.source_block['text'] = self.text
            self.source_block['category'] = self.category

    
    def crop_from_image(self, image, save_path=None):
        """Extract the bbox region from an image"""
        x1, y1, x2, y2 = self.bbox
        self.crop_img = CropImage(image.crop((x1, y1, x2, y2)), x_offset=x1)
        if save_path:
            self.crop_img.save(save_path)
        return self.crop_img
    
    def __repr__(self):
        if self.new_level is not None:
            return f"SectionHeader(new level={self.new_level}, bbox={self.bbox}, height={self.bbox[3]-self.bbox[1]}, width={self.bbox[2]-self.bbox[0]}, text='{self.clean_text}')"
        return f"SectionHeader(level={self.level}, bbox={self.bbox}, height={self.bbox[3]-self.bbox[1]}, width={self.bbox[2]-self.bbox[0]}, text='{self.clean_text}')"

class DirectoryStructure:
    def __init__(self):
        self.headers = []
    def add_header(self, info_block):
        header = SectionHeader.from_info_block(info_block)
        self.headers.append(header)
    
    def get_headers_by_level(self, level):
        return [h for h in self.headers if h.level == level]
    
    def get_all_headers(self):
        return self.headers
    
    def load_from_json(self, json_data):
        for info_block in json_data:
            if info_block.get('category') == 'Section-header' or info_block.get('category') == 'Title': #  or info_block.get('category') == 'List-item'  (Perhaps it is very long and recompute many time)
                self.add_header(info_block)

    def load_from_json_path(self, json_path):
        """Load Title, section headers and List-item from a JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.load_from_json(data["full_layout_info"])
    
    def extract_all_header_crops(self, image, save_dir=None):
        """Extract crops for all headers from an image"""
        crops = []
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for i, header in enumerate(self.headers):
            cropped_img = header.crop_from_image(image)
            crops.append(cropped_img)
            
            if save_dir:
                sanitized_text = re.sub(r'[\\/*?:"<>|\n\r]', '_', header.clean_text[:20])
                filename = f"header_{i}_level{header.level}_{sanitized_text}.png"
                save_path = os.path.join(save_dir, filename)
                cropped_img.image.save(save_path)
                print(f"Saved: {save_path}")
        
        return crops
    
    def __repr__(self):
        return f"DirectoryStructure({len(self.headers)} headers)"
    



class Reranker():

    def __init__(self, upper_level=1, now_level=MAX_LEVEL, sum_height=0, max_width=0):
        self.upper_level = upper_level
        self.now_level = now_level # current highest level in this batch
        self.sum_height = sum_height
        self.max_width = max_width
        self.highest_list = []
        self.check_list = []

        self.count = 0
        os.makedirs("/dots.ocr/test/output", exist_ok=True)
        self.parser = PageParser()

    def clear(self, upper_level=None):
        if upper_level is None:
            upper_level = self.now_level + 1
        self.upper_level = upper_level
        self.now_level = MAX_LEVEL
        self.sum_height = 0
        self.max_width = 0
        self.highest_list = []
        self.check_list = []

    def identify_highest_headers(self, cells, y_offset_list):

        print("---------------------------------------------------")

        highest_level = MAX_LEVEL
        highest_headers_idx = []
        h_idx = 0
        highest_list_is_empty = True if len(self.highest_list) == 0 else False

        for i, header in enumerate(self.check_list):
            print(y_offset_list[i], header)
        for info_block in cells["full_layout_info"]:
            print(info_block)

        for info_block in cells["full_layout_info"]:
            # print(info_block)
            now_header = SectionHeader.from_info_block(info_block)

            match_header_idx = []
            while h_idx < len(y_offset_list) and max(0, y_offset_list[h_idx] - info_block["bbox"][3]) / (y_offset_list[h_idx] - (0 if h_idx==0 else y_offset_list[h_idx-1])) < 0.33:
                print(y_offset_list[h_idx], info_block["bbox"][3])
                match_header_idx.append(h_idx)
                h_idx += 1
                
            print(f"highest_level: {highest_level}     level: {now_header.level}")
            if now_header.level < highest_level:
                highest_level = now_header.level
                highest_headers_idx = []
            if now_header.level == highest_level:
                highest_headers_idx.extend(match_header_idx)
                
            # print(highest_headers_idx)
        
        if not highest_list_is_empty and 0 not in highest_headers_idx:
            self.highest_list = []

        print(highest_headers_idx)
        new_check_list = []
        SAVE_NUM = 10 
        for i in range(len(highest_headers_idx)):
            if i < len(highest_headers_idx) - SAVE_NUM:
                self.highest_list.append(self.check_list[highest_headers_idx[i]])
            else:   
                new_check_list.append(self.check_list[highest_headers_idx[i]])
        self.check_list = new_check_list

    async def start_rerank(self):
        merge_cops = []
        for header in self.check_list:
            merge_cops.append(header.crop_img)
            # print(header)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        cells, y_offset_list = await self.merge_crops_and_parse(merge_cops, save_dir=None, save_name=None)
        self.identify_highest_headers(cells, y_offset_list)


    def insert(self, header: SectionHeader):
        self.sum_height += header.bbox[3] - header.bbox[1]
        self.max_width = max(self.max_width, header.bbox[2])
        if header.level < self.now_level:
            self.now_level = max(self.upper_level, header.level)

        self.check_list.append(header)

    async def try_to_insert(self, header: SectionHeader):
        h = self.sum_height + header.bbox[3] - header.bbox[1]
        w = max(self.max_width, header.bbox[2])
        if  h * w > MAX_PIXELS or h > MAX_LENGTH:
            await self.start_rerank()
            self.sum_height = 0
            self.max_width = 0
        self.insert(header)


    def assign_new_level(self):
        print(len(self.highest_list),len(self.check_list), self.now_level)
        for header in self.highest_list:
            header.new_level = self.now_level
        for header in self.check_list:
            header.new_level = self.now_level

            
    async def merge_crops_and_parse(self, crops, save_dir=None, save_name=None):
        if not crops:
            print("No crops provided")
            return None
        
        total_width = max(crop.image.width + crop.x_offset for crop in crops)
        total_height = sum(crop.image.height for crop in crops)
        
        merged_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        
        y_offset_list = []
        y_offset = 0
        for crop in crops:
            merged_image.paste(crop.image, (crop.x_offset, y_offset))
            y_offset += crop.image.height
            y_offset_list.append(y_offset)
        
        save_dir = f"test/output/output{self.count}"
        save_name = f"output{self.count}"
        print(f"Merged {len(crops)} crops into image of size {total_width}x{total_height}")
        assert(MAX_PIXELS >= total_width * total_height)

        try:
            if save_dir:
                result_debug = await self.parser._parse_single_image(
                    merged_image,
                    prompt_mode="prompt_layout_all_en",
                    save_dir=save_dir,
                    save_name=save_name
                )
            result = await self.parser._parse_single_image(
                merged_image,
                prompt_mode="prompt_layout_all_en",
                save_dir=None,
                save_name=None,
            )
            
            print(f"OCR parsing completed for merged crops")
            merged_image.save(f"test/output/output{self.count}/merged_image.jpg")
            self.count += 1
            return result, y_offset_list
            
        except Exception as e:
            print(f"Error during OCR parsing: {e}")
            return None


class DirectoryCleaner:
    def __init__(self):
        self.reranker = Reranker()
    
    async def reset_header_level(self, cells_list, images_origin):

        assert(len(cells_list) == len(images_origin))
        directorys = []
        all_sorted_indices = []
        for i, page in enumerate(cells_list):
            dir_structure = DirectoryStructure()
            dir_structure.load_from_json(page["full_layout_info"])
            dir_structure.extract_all_header_crops(image=images_origin[i])
            directorys.append(dir_structure)
        
            sorted_indices = sorted(range(len(dir_structure.headers)), key=lambda i: dir_structure.headers[i].level)
            all_sorted_indices.append(sorted_indices)
        
        begin = [0] * len(directorys)

        self.reranker.clear(upper_level=1)
        LEVEL_NUM = 8
        while self.reranker.upper_level <= LEVEL_NUM:
            print(f"llalalalal  {self.reranker.upper_level}")
            for i, sorted_indices in enumerate(all_sorted_indices):
                dir_structure = directorys[i]
                j = begin[i]
                headers = dir_structure.get_all_headers()
                now_level_in_this_page = -1
                
                allow = 2 
                while j < len(sorted_indices):
                    if now_level_in_this_page == -1:
                        now_level_in_this_page = headers[sorted_indices[j]].level
                    
                    if headers[sorted_indices[j]].level < now_level_in_this_page:
                        allow -= 1
                        now_level_in_this_page = headers[sorted_indices[j]].level
                        if allow < 0:
                            break
                    # print(headers[sorted_indices[j]])
                    await self.reranker.try_to_insert(headers[sorted_indices[j]])

                    j += 1
                print(f"========================{i}")

            if len(self.reranker.check_list) == 0:
                break
            
            await self.reranker.start_rerank()
            self.reranker.assign_new_level()
            self.reranker.clear()

            # elimite the headers that have been assigned new_level (might not be a continuous subsequence)
            for i, sorted_indices in enumerate(all_sorted_indices):
                dir_structure = directorys[i]
                begin[i] = 0
                new_sorted_indices = []
                for j in range(len(sorted_indices)):
                    if dir_structure.headers[sorted_indices[j]].new_level is None:
                        new_sorted_indices.append(sorted_indices[j])
                all_sorted_indices[i] = new_sorted_indices

        
        for dir_structure in directorys:
            for header in dir_structure.get_all_headers():
                if header.new_level is None:
                    header.new_level = 8
                header._reset_text_and_update()
                
                print(f"{header.text}")
                # print(f"{'#'*header.new_level} {header.clean_text}")
                # print(f"  {header}")
            print('----------------------')
