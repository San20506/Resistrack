import zipfile
import xml.dom.minidom

def get_docx_text(path):
    try:
        document = zipfile.ZipFile(path)
        xml_content = document.read('word/document.xml')
        document.close()
        dom = xml.dom.minidom.parseString(xml_content)
        texts = dom.getElementsByTagName('w:t')
        return ' '.join([t.firstChild.nodeValue for t in texts if t.firstChild])
    except Exception as e:
        return f"Error reading {path}: {str(e)}"

print('--- PRD ---')
print(get_docx_text('/home/sandy/Projects/Resistrack/ResisTrack_PRD.docx'))
print('\n--- Tech Arch ---')
print(get_docx_text('/home/sandy/Projects/Resistrack/ResisTrack_TechArch.docx'))
