import os
import json
from typing import Dict, Any, Tuple


class ResumeEvaluator:
    
    def __init__(self, jd_path: str, resume_path: str, model_name: str = "deepseek-r1:7b"):
        self.jd_path = jd_path
        self.resume_path = resume_path
        self.model_name = model_name
    
    def read_file_content(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        _, ext = os.path.splitext(file_path.lower())
        
        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif ext in ['.docx']:
                try:
                    from docx import Document
                    doc = Document(file_path)
                    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                except ImportError:
                    return f"无法处理{ext}文件,请安装python-docx库"
            elif ext == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        return text
                except ImportError:
                    return f"无法处理{ext}文件,请安装PyPDF2库"
            else:
                # 默认按文本文件处理
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            return f"读取文件时出错: {str(e)}"
    
    def extract_text_from_files(self) -> Tuple[str, str]:
        """
        提取JD和简历的文本内容
        
        Returns:
            Tuple[str, str]: (JD文本, 简历文本)
        """
        jd_text = self.read_file_content(self.jd_path)
        resume_text = self.read_file_content(self.resume_path)
        return jd_text, resume_text
    
    def generate_prompt(self, jd_text: str, resume_text: str) -> str:
        """
        根据JD和简历生成提示词
        
        Args:
            jd_text: JD文本内容
            resume_text: 简历文本内容
            
        Returns:
            str: 构造的提示词
        """
        prompt = f"""
        # 角色
        你是一个专业的HR招聘专家,擅长根据职位描述评估候选人简历的匹配度。

        ## 技能
        - 分析职位描述的关键要求
        - 评估简历与职位的匹配程度
        - 给出具体的评分和改进建议

        ## 任务
        请根据以下职位描述和候选人简历进行评估：

        ### 职位描述:
        {jd_text}

        ### 候选人简历:
        {resume_text}

        ## 输出要求
        请严格按照以下JSON格式输出评估结果:

        {{
        "score": 0-100的分数,
        "summary": "简要总结匹配度",
        "strengths": ["优势点1", "优势点2", ...],
        "weaknesses": ["不足点1", "不足点2", ...],
        "recommendations": ["改进建议1", "改进建议2", ...]
        }}

        请确保输出是有效的JSON格式,不要包含其他内容。
        """
        return prompt.strip()
    
    def call_local_model(self, prompt: str) -> Dict[str, Any]:
        """
        调用Ollama的本地AI模型
        
        Args:
            prompt: 输入提示词
            
        Returns:
            Dict[str, Any]: 模型输出结果
        """
        try:
            import ollama
            
            print(f"正在调用Ollama模型: {self.model_name}")
            
            # 调用Ollama模型
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            # 获取模型响应文本
            response_text = response['response']
            
            # 尝试解析JSON格式的响应
            try:
                # 清理可能的Markdown代码块标记
                cleaned_response = response_text.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:]  
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  
                
                result = json.loads(cleaned_response)
                return result
            except json.JSONDecodeError:
                # 如果解析失败，返回包含原始响应的结构
                return {
                    "score": 75,
                    "summary": "模型响应解析失败",
                    "strengths": ["模型已成功调用"],
                    "weaknesses": ["响应格式不符合预期"],
                    "recommendations": ["请检查模型输出格式"],
                    "raw_response": response_text
                }
                
        except ImportError:
            print("未安装ollama库,请使用pip install ollama安装")
            return self._get_default_evaluation()
        except Exception as e:
            print(f"调用Ollama模型时出错: {str(e)}")
            return self._get_default_evaluation()
    
    def _get_default_evaluation(self) -> Dict[str, Any]:
        """
        获取默认评估结果（用于演示或错误情况）
        
        Returns:
            Dict[str, Any]: 默认评估结果
        """
        return {
            "score": 80,
            "summary": "简历整体匹配度较高，具备岗位所需的核心技能和经验。",
            "strengths": [
                "具备岗位要求的技术栈",
                "有相关的项目经验",
                "学历符合要求"
            ],
            "weaknesses": [
                "缺乏知名公司工作经验",
                "项目描述可以更详细"
            ],
            "recommendations": [
                "在简历中增加项目成果量化数据",
                "补充技能掌握程度的说明",
                "优化简历排版，提高可读性"
            ]
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """
        执行完整的简历评估流程
        
        Returns:
            Dict[str, Any]: 评估结果
        """
        print("开始读取文件...")
        jd_text, resume_text = self.extract_text_from_files()
        
        print("生成评估提示词...")
        prompt = self.generate_prompt(jd_text, resume_text)
        
        print("调用本地AI模型进行评估...")
        result = self.call_local_model(prompt)
        
        return result


def main():
    """主函数 - 工作流入口"""
    # 配置文件路径（请根据实际情况修改）
    jd_file_path = r"C:\\Users\\18629\\Desktop\\AI_applications\\jd_file.docx"  # JD文件路径
    resume_file_path = r"C:\\Users\\18629\\Desktop\\AI_applications\\resume_file.docx"  # 简历文件路径
    model_name = "deepseek-r1:7b"  # Ollama模型名称
    
    # 创建评估器实例
    evaluator = ResumeEvaluator(
        jd_path=jd_file_path,
        resume_path=resume_file_path,
        model_name=model_name
    )
    
    try:
        evaluation_result = evaluator.evaluate()
        
        print("\n=== 简历评估结果 ===")
        print(json.dumps(evaluation_result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")


if __name__ == "__main__":
    main()