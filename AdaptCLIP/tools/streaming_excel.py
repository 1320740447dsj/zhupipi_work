"""Minimal constant-memory XLSX writer for streaming evaluation results."""

import math
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


_XML_HEADER = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def _column_name(index):
    name = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell_xml(row, column, value, header=False):
    reference = f'{_column_name(column)}{row}'
    style = ' s="1"' if header else ''
    if value is None:
        return f'<c r="{reference}"{style}/>'
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style}><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f'<c r="{reference}"{style}><v>{value}</v></c>'
    text = escape(str(value), {'"': '&quot;'})
    return f'<c r="{reference}" t="inlineStr"{style}><is><t>{text}</t></is></c>'


class _StreamingSheet:
    def __init__(self, path, headers, widths):
        self.path = path
        self.headers = headers
        self.row = 0
        self.file = open(path, 'w', encoding='utf-8', newline='')
        self.file.write(_XML_HEADER)
        self.file.write('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">')
        self.file.write('<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>')
        self.file.write('<cols>')
        for index, width in enumerate(widths, start=1):
            self.file.write(f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>')
        self.file.write('</cols><sheetData>')
        self.append(headers, header=True)

    def append(self, values, header=False):
        self.row += 1
        cells = ''.join(_cell_xml(self.row, index, value, header) for index, value in enumerate(values, start=1))
        self.file.write(f'<row r="{self.row}">{cells}</row>')

    def close(self):
        last_column = _column_name(len(self.headers))
        self.file.write('</sheetData>')
        self.file.write(f'<autoFilter ref="A1:{last_column}{self.row}"/>')
        self.file.write('</worksheet>')
        self.file.close()


class StreamingExcelWriter:
    """Append rows to temporary worksheet XML and finalize one XLSX file."""

    def __init__(self, output_path, sample_headers, metric_headers):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=self.output_path.parent)
        temp_root = Path(self.temp_dir.name)
        self.samples = _StreamingSheet(
            temp_root / 'samples.xml', sample_headers, [12, 18, 18, 55, 12, 16, 16, 16]
        )
        self.metrics = _StreamingSheet(
            temp_root / 'metrics.xml', metric_headers, [24] + [16] * (len(metric_headers) - 1)
        )
        self.closed = False

    def append_sample(self, values):
        self.samples.append(values)

    def append_metric(self, values):
        self.metrics.append(values)

    def close(self):
        if self.closed:
            return
        self.samples.close()
        self.metrics.close()
        with zipfile.ZipFile(self.output_path, 'w', compression=zipfile.ZIP_DEFLATED) as workbook:
            workbook.writestr('[Content_Types].xml', _CONTENT_TYPES)
            workbook.writestr('_rels/.rels', _ROOT_RELS)
            workbook.writestr('xl/workbook.xml', _WORKBOOK)
            workbook.writestr('xl/_rels/workbook.xml.rels', _WORKBOOK_RELS)
            workbook.writestr('xl/styles.xml', _STYLES)
            workbook.write(self.samples.path, 'xl/worksheets/sheet1.xml')
            workbook.write(self.metrics.path, 'xl/worksheets/sheet2.xml')
        self.temp_dir.cleanup()
        self.closed = True


_CONTENT_TYPES = _XML_HEADER + '''
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''

_ROOT_RELS = _XML_HEADER + '''
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

_WORKBOOK = _XML_HEADER + '''
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Samples" sheetId="1" r:id="rId1"/>
    <sheet name="Metrics" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>'''

_WORKBOOK_RELS = _XML_HEADER + '''
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

_STYLES = _XML_HEADER + '''
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>
  <fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''
